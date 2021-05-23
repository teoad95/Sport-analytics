import numpy as np
import cv2


color = (255, 150, 0)

def plot_bb_on_img(cv2_img, boxes):

    cv2_img_bb = np.array(cv2_img) 
    
    # Convert RGB to BGR 
    cv2_img_bb = cv2.cvtColor(cv2_img_bb, cv2.COLOR_BGR2RGB) 

    

    for i in range(len(boxes)):
            # extract the bounding box coordinates
            (x, y) = (int(boxes['xmin'][i]), int(boxes['ymin'][i]))
            (w, h) = (int(boxes['xmax'][i]-boxes['xmin'][i]), int(boxes['ymax'][i]-boxes['ymin'][i]))

            cv2.rectangle(cv2_img_bb, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(boxes['name'][i], boxes['confidence'][i])
            cv2.putText(cv2_img_bb, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)

    return cv2_img_bb