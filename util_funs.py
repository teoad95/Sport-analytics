import numpy as np
import cv2
import os
from PIL import Image
import math
import pandas as pd
import urllib
from pytube import YouTube
from sklearn.cluster import KMeans
import glob
 

color = (255, 150, 0)

def plot_bb_on_img(cv2_img, boxes, tolerance = 0.5, show_text=True):

    cv2_img_bb = np.array(cv2_img) 
    
    # Convert RGB to BGR 
    cv2_img_bb = cv2.cvtColor(cv2_img_bb, cv2.COLOR_BGR2RGB) 
    for i, _ in boxes.iterrows():
        if (boxes['confidence'][i] > tolerance): # filter predictions
            # extract the bounding box coordinates
            (x, y) = (int(boxes['xmin'][i]), int(boxes['ymin'][i]))
            (w, h) = (int(boxes['xmax'][i]-boxes['xmin'][i]), int(boxes['ymax'][i]-boxes['ymin'][i]))

            cv2.rectangle(cv2_img_bb, (x, y), (x + w, y + h), color, 2)
            if show_text:
                text = "{}: {:.4f}".format(boxes['name'][i], boxes['confidence'][i])
            else:
                text = ""
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
def keep_unique_objects_df(boxes_df, epsilon):
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


def split_image_and_predict(img, model, step_size=64, window_size=(256,256), epsilon=2):
    
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

    boxes_df = keep_unique_objects_df(boxes_df,epsilon)

    return boxes_df

def extract_average_color(img):
    # calculate the average color of each row of our image
    avg_color_per_row = np.average(img, axis=0)

    # calculate the averages of our rows
    avg_colors = np.average(avg_color_per_row, axis=0)
    # so, convert that array to integers
    int_averages = np.array(avg_colors, dtype=np.uint8)
    return int_averages

def download_video(url, path='Videos'):
    yt = YouTube(url)
    yt = yt.streams.filter(file_extension='mp4').first()
    out_file = yt.download(path)
    file_name = out_file.split("\\")[-1]
    print(f"Downloaded {file_name}, in location {path} correctly!")
    return file_name

def get_video_name(video_name_with_location):
    p = video_name_with_location.split("/")
    return p[-1].replace(".mp4", "")

def extract_images_from_video(video):    
    vidcap = cv2.VideoCapture('.//Videos//' + video)
    count = 0

    path = os.getcwd() + '/Images'
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + '/' + get_video_name(video)
    if not os.path.exists(path):
        os.mkdir(path)
        
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            write = cv2.imwrite(path + "/frame"+str(count)+".jpg", image)     # save frame as JPG file
        return hasFrames
    sec = 2
    frameRate = 2 #//it will capture image in each 0.5 second
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)

def get_frames_from_youtube_video(video_url):
    video = download_video(video_url)
    extract_images_from_video(video)
    os.remove(video)
    
def extract_average_color(img):
    # calculate the average color of each row of our image
    avg_color_per_row = np.average(img, axis=0)

    # calculate the averages of our rows
    avg_colors = np.average(avg_color_per_row, axis=0)
    # so, convert that array to integers
    int_averages = np.array(avg_colors, dtype=np.uint8)
    return int_averages

kmeans_trained = False
clusters = KMeans(2, random_state= 40)

def classify_players(img, boxes):
    global kmeans_trained
    players = []
    boxes.assign(Name='image')
    for i, b in boxes.iterrows():
        #boxes.insert(i, 'image', res_bgr[int(b['ymin']):int(b['ymax']), int(b['xmin']):int(b['xmax'])])
        players.append(img[int(b['ymin']):int(b['ymax']), int(b['xmin']):int(b['xmax'])])
    boxes['image'] = players
    features = []
    features = [extract_average_color(b.image) for i,b in boxes.iterrows()]
    if (not kmeans_trained):
        clustering_results = clusters.fit_predict(features)
        kmeans_trained = True
    else:
        clustering_results = clusters.predict(features)
    boxes['team'] = clustering_results
    return boxes

def plot_bb_on_img_with_teams(cv2_img, boxes, tolerance = 0.5, show_text=True):

    cv2_img_bb = np.array(cv2_img) 
    
    # Convert RGB to BGR 
    cv2_img_bb = cv2.cvtColor(cv2_img_bb, cv2.COLOR_BGR2RGB) 
    for i, _ in boxes.iterrows():
        if (boxes['confidence'][i] > tolerance): # filter predictions
            # extract the bounding box coordinates
            (x, y) = (int(boxes['xmin'][i]), int(boxes['ymin'][i]))
            (w, h) = (int(boxes['xmax'][i]-boxes['xmin'][i]), int(boxes['ymax'][i]-boxes['ymin'][i]))
            color = (255,0,0)
            if (boxes['team'][i] == 1):
                color = (0,255,0)
            cv2.rectangle(cv2_img_bb, (x, y), (x + w, y + h), color, 2)
            if show_text:
                text = "{}: {:.4f}".format(boxes['name'][i], boxes['confidence'][i])
            else:
                text = ""
            cv2.putText(cv2_img_bb, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
    return cv2_img_bb





def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))
