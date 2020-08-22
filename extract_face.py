import os 
import cv2
import numpy as np
from tqdm import tqdm
from utils import face_detection

test_data_path = "test_data1"

imagePaths = []

for i in os.listdir(test_data_path):
    imagePaths.append(f"{test_data_path}/{i}")

for (i,imagePath) in tqdm(enumerate(imagePaths),total=len(imagePaths)):
    image = cv2.imread(imagePath)
    bbox = face_detection(imagePath)
    startX,startY,endX,endY = bbox.astype('int')
    face = image[startY:endY,startX:endX]
    cv2.rectangle(image,(startX,startY),(endX,endY),(0,255,0),1)
    cv2.imshow('image',image)
    cv2.waitKey(0)



