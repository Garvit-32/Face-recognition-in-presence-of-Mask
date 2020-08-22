import os
import numpy as np
import pandas as pd
import cv2
from fastai import *
from fastai.vision import *
import torch
from PIL import Image


learn = load_learner(".")


classNames = ['angela_markel','anushka_sharma','donald_trump','narendra_modi',"salman_khan",'shushant_singh_rajput',"valdimir_putin"]
# for i in os.listdir('dataset'):
#     classNames.append(i)


img = open_image('test_data1/imdasd.jpeg')
# (image.jpg is any random image.)
img.show(figsize=(3, 3))
pred_class,preds_idx,outputs = learn.predict(img)
print(pred_class)
# learn.predict(dataset_with_mask_face/ellie_satler/00000006.jpg)


print(classNames[preds_idx])