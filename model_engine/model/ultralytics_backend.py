from ..util.log import logger
from ..util.redis_interface import redis_get, redis_set, redis_delete

from PIL import Image
import requests
from io import BytesIO
import random
from pathlib import Path
import yaml
import json
import numpy as np

from label_studio_ml.model import LabelStudioMLBase, LABEL_STUDIO_ML_BACKEND_V2_DEFAULT
from label_studio_ml.utils import get_single_tag_keys

from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes, Results
from ultralytics.yolo.data import YOLODataset

LABEL_STUDIO_ML_BACKEND_V2_DEFAULT = True

RECT_LABEL_TYPE = 'RectangleLabels'
IMG_TYPE = 'Image'

# LABEL_CONFIG_FILE = '.data/label_config.xml'
# MODEL_FILE = '.data/best.pt'
SEED_MODEL = 'models/yolov8n.pt'

LS_HOSTNAME = 'http://cacplcemudev.corp.hatchglobal.com:8080'

TRAIN_VAL_TEST_WEIGHT = (85, 15, 0)
TRAIN_EPOCHS = 2

DATASET_DIR = 'datasets'
MODEL_DIR = 'models'

class LabelStudioResultValue:
    def __init__(self, value: dict, cls_dict: dict):
        self.width = value['width'] / 100
        self.height = value['height'] / 100
        self.centre_x = value['x'] / 100 + self.width / 2  # add half the width to get the centre
        self.centre_y = value['y'] / 100 + self.height / 2  # add half the height to get the centre
        self.cls = cls_dict[value['rectanglelabels'][0]]

    @property
    def annotation(self):
        return f'{self.cls} {self.centre_x} {self.centre_y} {self.width} {self.height}'


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
        logger.info(f'Creating a new model object with kwargs: {kwargs}')
        kwargs['hostname'] = LS_HOSTNAME

        if 'access_token' in kwargs:
            logger.debug(f'...storing the access token for later use: {kwargs["access_token"]}')
            self.__class__.ACCESS_TOKEN = kwargs['access_token']  # store the access token class-wide... seems it does not get passed when creating a new instance for training (limits access to images)
        else:
            logger.debug(f'...stealing the access token from the class: {self.__class__.ACCESS_TOKEN}')
            kwargs['access_token'] = self.__class__.ACCESS_TOKEN
        
        super().__init__(model_dir=MODEL_DIR, **kwargs)
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config,
            RECT_LABEL_TYPE,
            IMG_TYPE
        )
        logger.info(f'{self} local path is: {self.get_local_path('')}')
        self.model = YOLO('models/yolov8n.pt')
        # if the mdoel has updates from training... load it!
        if self.train_output:
            logger.info(f'loading trained model: {self.train_output}')
            self.model.load(self.train_output['model_path'])
        else:
            logger.warning(f'could not locate any previously trained models... starting from scratch with: {SEED_MODEL}')
    
    def _get_image(self, task):
        url = task['data'][self.value]
        logger.debug(f'...getting image: {url}')
        img = None
        img_loc = redis_get(url)
        if img_loc is not None:
            if Path(img_loc).exists():
                logger.info('...found local copy of the image {loc_img} using that!')
                img = Image.open(img_loc)
                return img, Path(img_loc)
            else:
                logger.warning('...image path stored in redis, but could not be located here... removing the key')
                redis_delete(url)

        response = self.api_get(url)
        img = Image.open(BytesIO(response.content))
        return img, Path(url)
    
    def api_get(self, url):
        headers = {'Authorization': 'Token ' + self.access_token}
        logger.debug(f'...requesting data from: {self.hostname + url} with headers {headers}')
        return requests.get(self.hostname + url, stream=True, headers=headers)
    
    def api_get_json(self, url):
        return json.loads(self.api_get(url).content)

    def predict(self, tasks, **kwargs):
        logger.info('##### PREDICT CALL! #####')
        # model = YOLO(model_file)
        predictions = []
        logger.info(f'Total number of tasks received: {len(tasks)}')
        for task in tasks:
            print(f'Task: {task}')
            if 'frame' in kwargs:
                img = kwargs['frame']
            else:
                img = self._get_image(task)[0]

            results = self.model(img)[0]
            pred = YOLOResults(result=results, from_name=self.from_name, to_name=self.to_name)
            predictions.append(pred.prediction)
        return predictions

    def fit(self, tasks, workdir=None, ds_split=TRAIN_VAL_TEST_WEIGHT, epochs=TRAIN_EPOCHS, dataset=None, **kwargs):
        logger.info('##### TRAINING CALL! #####')
        logger.info(f'Fit data working directory: {workdir}')
        task_list = list(tasks)
        
        if self.train_output is not None:  #TODO: handle this more cleanly -- perhaps a task list hash map to dataset?
            dataset = self.train_output['data']['yaml_file']  # if no new tasks are submitted... just re-train on the old data for now...
        elif dataset is None:
            if len(task_list) > 0:
                dataset_path = Path(DATASET_DIR, Path(workdir).name)
                dataset = self.create_dataset(dataset_path=dataset_path, task_list=task_list, names=[], ds_split=ds_split)
            else:
                logger.warning('...exiting, no tasks or dataset provided to train on...')
                return

        self.model.train(data=str(dataset), epochs=epochs, project=workdir)
        train_result = {
            'checkpoints': self.model.trainer.wdir.as_posix(),
            'model_path': self.model.trainer.best.as_posix(),
            'epochs': self.model.trainer.epochs,
            'batch_size': self.model.trainer.batch_size,
            'metrics': self.model.trainer.metrics,
            'data': self.model.trainer.data
        }
        logger.info(f'Training completed with the following result: {train_result}')
        return train_result
    

    def create_dataset(self, dataset_path: Path, task_list, names = [], ds_split = TRAIN_VAL_TEST_WEIGHT):
        hash = hash(tuple(task_list))
        logger.info(f'searching local resources with dataset hash: {hash}')
        hash_str = f'ds_{hash}'
        ds_yml_path = redis_get(hash_str)
        if (ds_yml_path is not None) and Path(ds_yml_path).exists():
            return ds_yml_path
        
        logger.info(f'could not locate dataset with matching task hash, creating a new dataset: {dataset_path}')
        if not dataset_path.exists():
            dataset_path.mkdir()

        train_path = Path(dataset_path, 'train')
        val_path = Path(dataset_path, 'val')
        test_path = Path(dataset_path, 'test')
        
        cls_dict = {}
        # check the names the model is expecting first...
        for i, n in enumerate(names):
            cls_dict[n] = i  # assign a cls_id from zero, incrementing upwards
        # add any missing labels to our class dictionary
        for l in self.labels_in_config:
            if l not in cls_dict:
                cls_dict[l] = len(cls_dict)  # add the label to the dict with next sequential cls_id

        # form the dictionary for the dataset.yml
        ds_json = {
            'path': str(dataset_path.resolve()),
            'names': list(cls_dict.keys())
        }

        num_tasks = len(task_list)  # get the total number of tasks/images
        split_ratio = np.array(ds_split) / sum(ds_split)
        num_val = int(num_tasks * split_ratio[1]) # compute the number of validation images
        if len(split_ratio) < 3:  # if we don't even have a weight for the test split... ignore/treat as zero
            num_test = 0
        else:
            num_test = int(num_tasks * split_ratio[2])
        num_train = num_tasks - num_val - num_test  # training data is whatever is left over...

        logger.info(f'...will use random {num_train}/{num_tasks} for training, {num_val}/{num_tasks} for training validation, and {num_test}/{num_tasks} for final testing')
        
        random.shuffle(task_list)  # shuffle this bad boy up to ensure a good mix of images...

        for idx, t in enumerate(task_list):
            logger.info(f'[{idx}/{num_tasks}] processing for dataset: {dataset_path}')
            logger.debug(t)
            if idx < num_train:
                save_path = train_path
            elif idx < (num_train + num_val):
                save_path = val_path
            else:
                save_path = test_path

            if not save_path.name in ds_json:
                if not save_path.exists():
                    save_path.mkdir()
                ds_json[save_path.name] = save_path.name


            img, f = self._get_image(t)
            img_path = Path(save_path, f.name)
            img.save(str(img_path))  # save each image
            redis_set(f, str(img_path))
            
            annotations = []
            if 'annotations' in t:  # start parsing the annotations per image/task
                for a in t['annotations']:
                    if a['was_cancelled']:
                        print(f'Annotation [{a["id"]}] was cancelled... checking the next')
                        continue
                    for r in a['result']:
                        # print(r)
                        if 'value' in r:
                            v = LabelStudioResultValue(r['value'], cls_dict=cls_dict)
                            annotations.append(v.annotation + '\n')
                    break  # we already found a valid annotation list, stop looking... will maybe get duplicates(?)
            
            with open(str(Path(save_path, f.stem + '.txt')), 'w') as f:  # write the annotations to a txt file per image
                f.writelines(annotations)
        
        ds_yml_path = Path(dataset_path, 'dataset.yaml')
        with open(str(ds_yml_path), 'w') as yml:
            yaml.dump(ds_json, yml)

        redis_set(hash_str, ds_yml_path)
        return ds_yml_path  # return the path to the dataset we created...
    

    def get_project(self, task):
        if 'project' in task:
            proj = self.api_get_json(url = f'/api/projects/{task["project"]}/')
            logger.info(f'Got project info: {proj}')
            return proj
        
    
if __name__ == '__main__':
    pass
    
    # mod = YOLO(MODEL_FILE)
    # ds = Path('datasets/1679674886/dataset.yaml')
    # mod.train(data=str(ds.absolute()), project='models/1200L', epochs=TRAIN_EPOCHS)
