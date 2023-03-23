from PIL import Image
import requests
from io import BytesIO
import random
from pathlib import Path

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys

from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes, Results
from ultralytics.yolo.data import YOLODataset

RECT_LABEL_TYPE = 'RectangleLabels'
IMG_TYPE = 'Image'

LABEL_CONFIG_FILE = '.data/label_config.xml'
MODEL_FILE = '.data/best.pt'

LS_HOSTNAME = 'http://cacplcemudev.corp.hatchglobal.com:8080'


class YOLOResults:
    def __init__(self, result: Results, from_name: str, to_name: str, res_type=RECT_LABEL_TYPE):
        self.results = []
        self.scores = []
        self.rotation = 0
        self.type = 'rectanglelabels' #res_type.lower()
        self.names = result.names
        self.original_width, self.original_height = result.orig_shape
        id = 0
        for b in result.boxes:
            yr = YOLOResult(b, self.names)
            self.scores.append(yr.score)
            r = yr.result
            r['id'] = str(id)
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
            # 'model_version': self.model_version,
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
            'rotation': float(self.rotation),
            'x': float(self.x),
            'y': float(self.y),
            'width': float(self.width),
            'height': float(self.height),
            'rectanglelabels': [self.label]
        }
    
    @property
    def result(self):
        return {
            'score': self.score,
            'image_rotation': float(0),
            'value': self.value
        }

# good example here: https://github.com/heartexlabs/label-studio/discussions/1623#discussioncomment-1507940
class YOLOModel(LabelStudioMLBase):
    ACCESS_TOKEN = None
    def __init__(self, **kwargs):
        print(f'YOLO Model kwargs: {kwargs}')
        kwargs['hostname'] = LS_HOSTNAME
        if 'access_token' in kwargs:
            print(f'...storing the access token for later use: {kwargs["access_token"]}')
            self.__class__.ACCESS_TOKEN = kwargs['access_token']  # store the access token class-wide... seems it does not get passed when creating a new instance for training (limits access to images)
        else:
            print(f'...stealing the access token from the class: {self.__class__.ACCESS_TOKEN}')
            kwargs['access_token'] = self.__class__.ACCESS_TOKEN
        
        if 'label_config' not in kwargs:
            with open(LABEL_CONFIG_FILE, "r") as f:
                kwargs['label_config'] = f.read()
        super().__init__(**kwargs)
        # print(f'MLBase dir: {dir(self)}')
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config,
            RECT_LABEL_TYPE,
            IMG_TYPE
        )

    # def _get_image_url(self, task):
    #     url = self.hostname + task['data'][self.value]
    #     return url
    
    def _get_image(self, task):
        # headers = {'Authorization': 'Token ' + self.access_token}
        # response = requests.get(self._get_image_url(task), stream=True, headers=headers)
        url = task['data'][self.value]
        print(f'...getting image: {url}')
        response = self.api_get(url)
        try:
            img = Image.open(BytesIO(response.content))
        except Exception as e:
            print(e)
            print(f'response: {response}')
            print(f'raw response: {response.raw}')
            print(f'content: {response.content}')
        return img, Path(url).name
    
    def api_get(self, url):
        headers = {'Authorization': 'Token ' + self.access_token}
        print(f'...requesting data from: {self.hostname + url} with headers {headers}')
        return requests.get(self.hostname + url, stream=True, headers=headers)

    def predict(self, tasks, model_file=MODEL_FILE, **kwargs):
        model = YOLO(model_file)
        predictions = []
        print(f'Total number of tasks received: {len(tasks)}')
        for task in tasks:
            print(f'Task: {task}')
            if 'frame' in kwargs:
                img = kwargs['frame']
            else:
                img = self._get_image(task)[0]

            results = model(img)[0]
            pred = YOLOResults(result=results, from_name=self.from_name, to_name=self.to_name)
            predictions.append(pred.prediction)
        return predictions

    
    def fit(self, tasks, workdir=None, **kwargs):
        print('##### TRAINING CALL! #####')
        print(f'Fit tasks: {tasks} ({type(tasks)})')
        print(f'Fit workdir: {workdir}')
        print(f'Fit kwargs: {kwargs}')
        
        print('...preparing the dataset')
        dataset_dir = Path(workdir,'dataset')
        if not dataset_dir.exists():
            dataset_dir.mkdir()
        # for t in random.shuffle(tasks):
        for t in tasks:
            img, fn = self._get_image(t)
            print(f'got {fn}... ({type(img)}') #: {dir(img)})')
            img.save(str(Path(dataset_dir, fn)))
            for k, v in t.items():
                if k == 'annotations':
                    print(f'{k}:')
                    for li in v:
                        print(' [')
                        for k1, v1 in li.items():
                            print(f'    {k1}: {v1}')
                else:
                    print(f'{k}: {v}')
            

            # with open(Path(dataset_dir, fn), 'wb') as f:
            #     f.write(frame)
            # print(f'{t}')
            exit()
        
        img_path = ''

        # ds = YOLODataset(names=self.labels_in_config)
        print(f'Fit workdir: {workdir}')
        print(f'Fit kwargs: {kwargs}')

        # ... do some heavy computations, get your model and store checkpoints and resources
        return {'checkpoints': 'my/model/checkpoints'}  # <-- you can retrieve this dict as self.train_output in the subsequent calls

    
if __name__ == '__main__':
    TEST_KWARGS = {'label_config': '<View>\n  <Image name="image" value="$image"/>\n  <RectangleLabels name="label" toName="image">\n    <Label value="cage gate closed"/>\n    <Label value="cage gate opened"/>\n    <Label value="cage not at level"/>\n    <Label value="car"/>\n    <Label value="forklift"/>\n    <Label value="huddle"/>\n    <Label value="locomotive"/>\n    <Label value="muck"/>\n    <Label value="person"/>\n  </RectangleLabels>\n</View>', 'hostname': 'http://localhost:8080', 'access_token': 'e2ab97513d5aa0f91e9589ad949de1b3dbb297ad', 'model_version': None}
    TEST_TASK = {'id': 1, 'data': {'image': '/data/upload/1/781153e9-Craig_1200L_Collar_2023-02-19_05-00-001_0.png'}, 'meta': {}, 'created_at': '2023-03-21T13:27:24.481975Z', 'updated_at': '2023-03-21T13:27:24.482009Z', 'is_labeled': False, 'overlap': 1, 'inner_id': 1, 'total_annotations': 0, 'cancelled_annotations': 0, 'total_predictions': 0, 'comment_count': 0, 'unresolved_comment_count': 0, 'last_comment_updated_at': None, 'project': 1, 'updated_by': None, 'file_upload': 40, 'comment_authors': [], 'annotations': [], 'predictions': []}
    mod = YOLOModel(**TEST_KWARGS)
    results = mod.predict(tasks=[TEST_TASK], frame='.data/0.png')