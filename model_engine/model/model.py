from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys

from ultralytics import YOLO



# good example here: https://github.com/heartexlabs/label-studio/discussions/1623#discussioncomment-1507940
class YOLOModel(LabelStudioMLBase):
    def __init__(self, model_file: str, **kwargs):
        super().__init__(self, **kwargs)
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image'
        )
        self.model = YOLO(model_file)

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
            # for each task, return classification results in the form of "choices" pre-annotations
            # TODO: call the yolo model to return predictions....
            predictions.append({
                'result': [{
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'RectangleLabels',
                    'value': {'Label': ['My Label']}
                }],
                # optionally you can include prediction scores that you can use to sort the tasks and do active learning
                'score': 0.987
            })
        return predictions
    
    def fit(self, completions, workdir=None, **kwargs):
        # ... do some heavy computations, get your model and store checkpoints and resources
        return {'checkpoints': 'my/model/checkpoints'}  # <-- you can retrieve this dict as self.train_output in the subsequent calls
