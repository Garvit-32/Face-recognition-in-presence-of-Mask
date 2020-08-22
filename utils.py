import cv2
import torch
from face_detection.face_ssd_infer import SSD
from face_detection.utils import vis_detections
import os 
import sys
from tqdm import tqdm
from create_mask.mask import create_mask

def face_detection(img_path,device="cuda"):
    # device = torch.device("cuda")
    device = torch.device(device)
    conf_thresh = 0.3
    target_size = (300, 300)


    net = SSD("test")
    net.load_state_dict(torch.load('face_detection/weights/WIDERFace_DSFD_RES152.pth'))
    net.to(device).eval();

    # img_path = './imgs/12_Group_Group_12_Group_Group_12_128.jpg'
    # img_path = './test.png'

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # print("shape2afterimg ",img.shape)
    # print(img.shape)
    # detections = net.detect_on_image(img, size, device, is_pad=False, keep_thresh=conf_thresh)
    detections = net.detect_on_image(img, target_size, device, is_pad=False, keep_thresh=conf_thresh)
    bbox = vis_detections(img, detections, conf_thresh)#, show_text=False
    # print("shape3",img.shape)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)

    return bbox
    # print(x)
    # print(detections)

    # cv2.imshow('image',img)
    # cv2.waitKey(0)
# image = cv2.imread("dataset_with_mask/ellie_sattler/00000054.jpg")
# print("shape1 " ,image.shape)

# bbox = face_detection("dataset_with_mask/ellie_sattler/00000054.jpg")
# print(bbox)
# print(bbox)