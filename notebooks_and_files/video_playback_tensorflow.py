import cv2
from model import get_frame_with_predictions, add_Kmeans_features, train_Kmeans

video_filepath = './Videos/short2.mp4'

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

print('Started getting data for KMEANS')
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    cv2_img = frame[:, :, ::-1]  # OpenCV image (BGR to RGB) 
    ready_to_train_KMeans = add_Kmeans_features(cv2_img)
  else: 
    break
  if (ready_to_train_KMeans):
    print('Ready to train KMEANS')
    break
print('Finished getting data for KMEANS')
try:
  # When everything done, release the video capture object
  cap.release()
  out.release()
  # Closes all the frames
  cv2.destroyAllWindows()
except:
  print('Saved')
print('Train KMEANS')
train_Kmeans()

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
print('Start video')
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    cv2_img = frame[:, :, ::-1]  # OpenCV image (BGR to RGB) 

    # plot bounding boxes
    cv2_img_bb = get_frame_with_predictions(cv2_img)

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
