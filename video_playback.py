import cv2
import torch
from PIL import Image
from util_funs import plot_bb_on_img, split_image_and_predict


SINGLE_FRAME_INFER = False


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

video_filepath = './Videos/video1.mp4'

cap = cv2.VideoCapture(video_filepath)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    cv2_img = frame[:, :, ::-1]  # OpenCV image (BGR to RGB) 


    # inference
    if SINGLE_FRAME_INFER:
      results = model(cv2_img, size=320)  # includes NMS
      boxes = results.pandas().xyxy[0]
    else:
      boxes =  split_image_and_predict(cv2_img, model)

      
    # plot bounding boxes
    cv2_img_bb = plot_bb_on_img(cv2_img, boxes, tolerance=0.3)

    
    # Display the resulting frame
    cv2.imshow('Frame',cv2_img_bb)

    # Write the frame into the file 'output.avi'
    #out.write(cv2_img_bb)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()