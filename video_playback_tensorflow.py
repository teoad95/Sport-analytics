import cv2
from model import get_frame_with_predictions

video_filepath = './Videos/video1.mp4'

cap = cv2.VideoCapture(video_filepath)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(32)
frame_height = int(32)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

kmeans_trained = False
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    cv2_img = frame[:, :, ::-1]  # OpenCV image (BGR to RGB) 
    cv2_img_bb = get_frame_with_predictions(cv2_img, kmeans_trained)
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
out.release()

# Closes all the frames
cv2.destroyAllWindows()