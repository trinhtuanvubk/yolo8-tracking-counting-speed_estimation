from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model
import numpy as np 

yolo_nas = models.get("yolo_nas_s", checkpoint_path="./weights/yolo_nas_s_coco.pth", pretrained_weights="coco")
# model = models.get(model_name=Models.RESNET18, num_classes=10)
# load_checkpoint_to_model(net=model, ckpt_local_path="/path/to/my_checkpoints_folder/my_resnet18_training_experiment/ckpt_best.pth")
# s m l

# image_processor = ComposeProcessing(
#     [
#         DetectionLongestMaxSizeRescale(output_shape=(636, 636)),
#         DetectionCenterPadding(output_shape=(640, 640), pad_value=114),
#         StandardizeImage(max_value=255.0),
#         ImagePermute(permutation=(2, 0, 1)),
#     ]
# )
class_names = [
    "person",
    "bicycle",
    "car",
    "truck",
    "motorcycle",
    "airplane",
    "bus",
]
yolo_nas.set_dataset_processing_params(
    class_names=class_names,
    # image_processor=image_processor,
    iou=0.35, conf=0.25,
)

# class_names, prediction, image, 
# predic tion: _images_prediction_lst, fps
image_prediction = list(yolo_nas.predict("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg"))[0]
# a = list(yolo_nas.predict("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg"))[0].save("yolonastest_img.jpg")

# print(type(a))
# print(a['prediction'])

class_names = image_prediction.class_names
labels = image_prediction.prediction.labels
confidence = image_prediction.prediction.confidence
bboxes = image_prediction.prediction.bboxes_xyxy
print(bboxes)
print(len(bboxes))
print(labels)
print(len(labels))


array1 = np.array(labels).reshape((len(labels), 1))
array2 = np.array(bboxes)

print(array1.shape, array2.shape)
output = np.concatenate((array2, array1), axis=-1)
print(output)