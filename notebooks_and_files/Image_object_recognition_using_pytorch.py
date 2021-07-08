import cv2
import torch
from PIL import Image
from util_funs import plot_bb_on_img, get_predicted_objects_from_image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# generate a dataset from video using the generate_dataset notebook, and use any frame
image_path = './Images/Highlights Real Madrid vs FC Barcelona (2-1)/frame11.jpg'

img = cv2.imread(image_path)
results = model(img, size=320)  # includes NMS
boxes = results.pandas().xyxy[0]
get_predicted_objects_from_image(img, boxes)