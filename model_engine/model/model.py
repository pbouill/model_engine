from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys

from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes, Results

RECT_LABEL_TYPE = 'RectangleLabels'
IMG_TYPE = 'Image'

LABEL_CONFIG_FILE = '.data/label_config.xml'


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
    
"""
## Example prediction json format
[{
  "data": {
    "image": "/static/samples/sample.jpg" 
  },

  "predictions": [{
    "model_version": "one",
    "score": 0.5,
    "result": [
      {
        "id": "result1",
        "type": "rectanglelabels",        
        "from_name": "label", "to_name": "image",
        "original_width": 600, "original_height": 403,
        "image_rotation": 0,
        "value": {
          "rotation": 0,          
          "x": 4.98, "y": 12.82,
          "width": 32.52, "height": 44.91,
          "rectanglelabels": ["Airplane"]
        }
      },
      {
        "id": "result2",
        "type": "rectanglelabels",        
        "from_name": "label", "to_name": "image",
        "original_width": 600, "original_height": 403,
        "image_rotation": 0,
        "value": {
          "rotation": 0,          
          "x": 75.47, "y": 82.33,
          "width": 5.74, "height": 7.40,
          "rectanglelabels": ["Car"]
        }
      },
      {
        "id": "result3",
        "type": "choices",
        "from_name": "choice", "to_name": "image",
        "value": {
          "choices": ["Airbus"]
      }
    }]
  }]
}]
"""


# good example here: https://github.com/heartexlabs/label-studio/discussions/1623#discussioncomment-1507940
class YOLOModel(LabelStudioMLBase):
    def __init__(self, model_file: str, **kwargs):
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

        #     print(preds)
        #     for r in preds.pandas().xyxy[0].rows:
        #         predictions.append(
        #             {
        #                 'result': [
        #                     {
        #                         'from_name': from_name,
        #                         'to_name': to_name,
        #                         'type': 'RectangleLabels',
        #                         'value': {
        #                             'Label': [
        #                                 'My Label'
        #                             ]
        #                         }
        #                     }
        #                 ],
        #                 # optionally you can include prediction scores that you can use to sort the tasks and do active learning
        #                 'score': 0.987
        #             }
        #         )
        # return predictions
    
    def fit(self, completions, workdir=None, **kwargs):
        # ... do some heavy computations, get your model and store checkpoints and resources
        return {'checkpoints': 'my/model/checkpoints'}  # <-- you can retrieve this dict as self.train_output in the subsequent calls
    
    # def _format_pred(self, pred):
    #     names = pred.names
    #     for b in pred.boxes:
    #         r = YOLOResultValue(b, names)
    #         print(r.value)

    
if __name__ == '__main__':
    mod = YOLOModel('.data/best.pt')
    # print(f'Model: {dir(mod.model)}')
    # print(f'model.model: {dir(mod.model.model)}')
    # exit()
    results = mod.model('.data/0.png')[0]
    preds = YOLOResults(results, mod.from_name, mod.to_name, mod.model.model.pt_path)
    print(preds.prediction)
    # print(f'predictions: {preds}')
    # print(f'Total of {len(preds)} predictions')

    # print(dir(preds))
    # print(f'predictions: {preds["predictions"].pandas().xyxy[0]}')
    # predictions = []
    # for p in preds:
    #     res = YOLOResults(p, mod.from_name, mod.to_name, mod.model.cfg)
    #     predictions.append(res.prediction)
        # print(res.result)
    #     # print(p, dir(p))
    #     # print(f'Pandas: {p.pandas()}')
    #     # print(f'Keys: {p.keys}')
        
    #     names = p.names
    #     # probs = p.probs
    #     # path = p.path

    #     print(f'Names: {names}')
        # print(f'Probs: {probs}')
        # print(f'Path: {path}')
        # for b in p.boxes:
            # print(b, dir(b))
            # print(f'Cls: {b.cls}')
            # print(f'ID: {b.id}')
            # print(f'Shape: {b.shape}')
            # print(f'Orig Shape: {b.orig_shape}')
            # print(f'XYXY: {b.xyxy}')
            # print(f'XYXYn: {b.xyxyn}')
            # print(f'XYWH: {b.xywh}')
            # print(f'XYWHn: {b.xywhn}')
            # exit()
            # print(f'Pandas: {b.pandas()}')
        # print(p.pandas())
        # mod._format_pred(p)
    # print(predictions)
    
