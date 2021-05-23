import cv2
import torch
from PIL import Image
from util_funs import plot_bb_on_img

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

video_filepath = './Videos/video1.mp4'

cap = cv2.VideoCapture(video_filepath)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    cv2_img = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)  
    # inference
    results = model(cv2_img, size=640)  # includes NMS
    boxes = results.pandas().xyxy[0]

    # plot bounding boxes
    cv2_img_bb = plot_bb_on_img(cv2_img, boxes)
    # Display the resulting frame
    cv2.imshow('Frame',cv2_img_bb)

    # Press Q on keyboard to  exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()