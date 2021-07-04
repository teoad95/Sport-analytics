import cv2
import numpy as np
from sklearn.cluster import KMeans
from util_funs import extract_color_histogram

class model:
    def __init__(self):
        # model trained to detect only persons
        weightsPath = "network_configuration/frozen_inference_graph.pb"
        configPath = "network_configuration/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        self.__net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
    
    def get_boxes_and_masks(self, image):
        # detect objects
        blob = cv2.dnn.blobFromImage(image, swapRB=True)
        self.__net.setInput(blob)
        return self.__net.forward(["detection_out_final", "detection_masks"])
        
__model = model()
__features_to_train_Kmeans = []
__required_number_of_features = 10
#__Kmeans_trained = False
__clusters = KMeans(2, random_state = 40)

def get_frame_with_predictions(image, kmeans_trained):
    cv2_img_bb = np.array(image) 
    # Convert RGB to BGR 
    cv2_img_bb = cv2.cvtColor(cv2_img_bb, cv2.COLOR_BGR2RGB) 
    height, width , _ = image.shape
    boxes, masks = __model.get_boxes_and_masks(image)
    for i in range(boxes.shape[2]):
        box = boxes[0, 0 ,i]
        class_id = box[1]
        score = box[2]
        if (score < 0.5):
            continue
        # get box coordinates
        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)

        object_of_image = image[y: y2, x: x2]
        ooi_height, ooi_width, _ = object_of_image.shape

        # get the mask
        mask = masks[i, int(class_id)]
        mask = cv2.resize(mask, (ooi_width, ooi_height))
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        # convert mask to proper type in order to perform the bitwise_and
        visMask = (mask * 255).astype("uint8")

        # extract from box only the object for classification
        object = cv2.bitwise_and(object_of_image, object_of_image, mask = visMask)
        kmeans_result = 0

        if (not kmeans_trained):
            if (len(__features_to_train_Kmeans) < __required_number_of_features):
                __features_to_train_Kmeans.append(extract_color_histogram(object))
            else:
                __clusters.fit(__features_to_train_Kmeans)
                kmeans_trained = True
        else:
            kmeans_result = __clusters.fit_predict(extract_color_histogram(object).reshape(1, -1))
    

        #kmeans_result = __clusters.fit_predict(extract_color_histogram(object).reshape(1, -1))
        team = 'team A'
        if (kmeans_result == 1):
            team = 'team B'
        print(kmeans_result + ' features to train:' + len(__features_to_train_Kmeans))
        cv2.rectangle(cv2_img_bb, (x,y), (x2,y2), (255, 0, 0), 3)
        cv2.putText(cv2_img_bb, team, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)
    return cv2_img_bb

            


