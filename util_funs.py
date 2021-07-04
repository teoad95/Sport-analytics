import numpy as np
import cv2
import os
from PIL import Image
import math
import pandas as pd


color = (255, 150, 0)

def plot_bb_on_img(cv2_img, boxes, tolerance = 0.5):

    cv2_img_bb = np.array(cv2_img) 
    
    # Convert RGB to BGR 
    cv2_img_bb = cv2.cvtColor(cv2_img_bb, cv2.COLOR_BGR2RGB) 
    for i, _ in boxes.iterrows():
        if (boxes['confidence'][i] > tolerance): # filter predictions
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


def sliding_window(img, step_size, window_size):
#     img = Image.open(input_file)
    
    img_width, img_height = img.size
    for y in range(0, img_height, step_size):
        for x in range(0, img_width, step_size):
            box = (x, y, x+window_size[0], y+window_size[1])
            
            if x+window_size[0] <= img_width  and y+window_size[1] <= img_height :
                yield (box, img.crop(box))
                
# *to change* 
# check if objects already detected before adding it to list
def keep_unique_objects_df(boxes_df, epsilon=2):
    ## Remove duplicate objects ##    
    distances = boxes_df.distance.to_list()

    for distance in distances:
        first = True

        for index, row in boxes_df.iterrows():
            if first and abs(row.distance-distance) < epsilon:
                first = False
                continue
            else:
                if abs(row.distance-distance) < epsilon:
                    boxes_df.drop(index, inplace=True)   
    return boxes_df


def split_image_and_predict(img, model, step_size=64, window_size=(256,256)):
    
    img = Image.fromarray(img[:,:,::-1])
    
    boxes_df = pd.DataFrame(columns = ['xmin','ymin','xmax','ymax','confidence','class','name','centerx','centery'])
    
    for i,(box,img) in enumerate(sliding_window(img, step_size, window_size)):
       
        # inference
        results = model(img, size=320)  # includes NMS
        boxes = results.pandas().xyxy[0]
 
        # project to starting image
        boxes.xmin = boxes.xmin+box[0]
        boxes.xmax = boxes.xmax+box[0]

        boxes.ymin = boxes.ymin+box[1]
        boxes.ymax = boxes.ymax+box[1]

        boxes['centerx'] = boxes.xmax - (boxes.xmax-boxes.xmin)/2
        boxes['centery'] = boxes.ymax - (boxes.ymax-boxes.ymin)/2
        boxes['distance'] = np.sqrt(boxes.centerx*boxes.centerx + boxes.centery*boxes.centery)

        boxes_df = boxes_df.append(boxes, ignore_index=True)

    boxes_df = keep_unique_objects_df(boxes_df)

    return boxes_df

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	return hist.flatten()