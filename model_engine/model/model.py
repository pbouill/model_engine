from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys

from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes, Results

RECT_LABEL_TYPE = 'RectangleLabels'
IMG_TYPE = 'Image'

LABEL_CONFIG_FILE = '.data/label_config.xml'
MODEL_FILE = '.data/best.pt'


class YOLOResults:
    def __init__(self, result: Results, from_name: str, to_name: str, model_version: str = None, res_type=RECT_LABEL_TYPE):
        self.results = []
        self.scores = []
        self.rotation = 0
        self.model_version = model_version
        self.type = 'rectanglelabels' #res_type.lower()
        self.names = result.names
        self.original_width, self.original_height = result.orig_shape
        id = 0
        for b in result.boxes:
            yr = YOLOResult(b, self.names)
            self.scores.append(yr.score)
            r = yr.result
            r['id'] = id
            r['from_name'] = from_name
            r['to_name'] = to_name
            r['type'] = res_type.lower()
            r['image_rotation'] = 0
            r['original_width'] = self.original_width
            r['original_height'] = self.original_height
            self.results.append(r)
            id += 1


    @property
    def prediction(self):
        return {
            'model_version': self.model_version,
            'result': self.results,
            'score': sum(self.scores) / len(self.scores)
        }


class YOLOResult:
    def __init__(self, box: Boxes, names: dict):
        self.rotation = 0
        self.x, self.y = box.xyxyn[0][0:2].numpy() * 100
        self.width, self.height = box.xywhn[0][2:4].numpy() * 100
        self.label = names[int(box.cls)]
        self.score = float(box.conf * 100)
    
    @property
    def value(self):
        return {
            'rotation': self.rotation,
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'rectanglelabels': [self.label]
        }
    
    @property
    def result(self):
        return {
            'score': self.score,
            'image_rotation': 0,
            'value': self.value
        }

# good example here: https://github.com/heartexlabs/label-studio/discussions/1623#discussioncomment-1507940
class YOLOModel(LabelStudioMLBase):
    def __init__(self, model_file: str = MODEL_FILE, **kwargs):
        self.model = YOLO(model_file)
        with open(LABEL_CONFIG_FILE, "r") as f:
            label_config = f.read()
        super().__init__(label_config=label_config, **kwargs)
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config,
            RECT_LABEL_TYPE,
            IMG_TYPE
        )
        print(f'From Name: {self.from_name}; To Name: {self.to_name}; Labels: {self.labels_in_config}')

    def _get_image_url(self, task):
        return task['data'][self.value]
    
    def _get_image_frame(self, task):
        return Image.open(self._get_image_url(task))

    def predict(self, tasks, **kwargs):
        predictions = []
        # Get annotation tag first, and extract from_name/to_name keys from the labeling config to make predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema['to_name'][0]
        for task in tasks:
            print(f'Task: {task}')
            print(dir(task))
            # for each task, return classification results in the form of "choices" pre-annotations
            # TODO: call the yolo model to return predictions....
            img = self._get_image_frame(task)
            results = self.model(img)[0]
            pred = YOLOResults(results=results, from_name=self.from_name, to_name=self.to_name, model_version=self.model.model.pt_path)
            predictions.append(pred)
        return predictions

    
    def fit(self, completions, workdir=None, **kwargs):
        # ... do some heavy computations, get your model and store checkpoints and resources
        return {'checkpoints': 'my/model/checkpoints'}  # <-- you can retrieve this dict as self.train_output in the subsequent calls

    
if __name__ == '__main__':
    mod = YOLOModel('.data/best.pt')

    results = mod.model('.data/0.png')[0]
    preds = YOLOResults(results, mod.from_name, mod.to_name, mod.model.model.pt_path)
    print(preds.prediction)