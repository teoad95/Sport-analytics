import numpy as np
import cv2
import os


color = (255, 150, 0)

def plot_bb_on_img(cv2_img, boxes):

    cv2_img_bb = np.array(cv2_img) 
    
    # Convert RGB to BGR 
    cv2_img_bb = cv2.cvtColor(cv2_img_bb, cv2.COLOR_BGR2RGB) 
    for i in range(len(boxes)):
        if (boxes['confidence'][i] > 0.6): # filter predictions
            # extract the bounding box coordinates
            (x, y) = (int(boxes['xmin'][i]), int(boxes['ymin'][i]))
            (w, h) = (int(boxes['xmax'][i]-boxes['xmin'][i]), int(boxes['ymax'][i]-boxes['ymin'][i]))

            cv2.rectangle(cv2_img_bb, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(boxes['name'][i], boxes['confidence'][i])
            cv2.putText(cv2_img_bb, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
    return cv2_img_bb

def get_predicted_objects_from_image(cv2_img, boxes):
    path = os.getcwd() + '/ImagesFromObject'
    if not os.path.exists(path):
        os.mkdir(path)
    
    for i in range(len(boxes)):
       a = boxes['name'][i] 
       if (boxes['name'][i] == 'person'):
           cropped_img = cv2_img[int(boxes['ymin'][i]) - 5: int(boxes['ymax'][i]) + 5, int(boxes['xmin'][i]) - 5: int(boxes['xmax'][i]) + 5]
           write = cv2.imwrite(path + "/person"+str(i)+".jpg", cropped_img)

