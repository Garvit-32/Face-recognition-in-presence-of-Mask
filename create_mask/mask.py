import os
import sys
import numpy as np
import face_recognition
import random
from PIL import Image, ImageFile
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# image_path = 'dataset/00000051.jpg'
# image_path = 'me.jpg'

# mask_path = os.path.join(os.getcwd(),"create_mask/mask.png")

# mask_list = []

# default_path = os.path.join(os.getcwd(),"create_mask/mask.png")
# white_path = os.path.join(os.getcwd(),"create_mask/white.png")
# # blue_path = os.path.join(os.getcwd(),"create_mask/blue.png")
# black_path = os.path.join(os.getcwd(),"create_mask/black.png")

# mask_list.append(default_path)
# mask_list.append(white_path)
# # mask_list.append(blue_path)
# mask_list.append(black_path)

# mask_path = random.choice(mask_list)
# print(mask_path)


def create_mask(image_path,mask_path):
    mask_path = mask_path
    pic_path = image_path
    # mask_path = mask_path
    key_facial_feature = ('nose_bridge', 'chin')

    # comment
    # cv_img = cv2.imread(pic_path)

    face_image_np = face_recognition.load_image_file(pic_path)
    # print(face_image_np)
    face_locations = face_recognition.face_locations(
        face_image_np, model='hog')
    face_landmarks = face_recognition.face_landmarks(
        face_image_np, face_locations)
    face_img = Image.open(pic_path)
    # face_img.show()
    mask_img = Image.open(mask_path)

    found_face = False
    for face_landmark in face_landmarks:

        skip = False

        for facial_feature in key_facial_feature:

            # comment
            # for i in face_landmark[facial_feature]:
            #     cv2.circle(cv_img, (i[0], i[1]), 2, (0, 0, 255), 2)
            # cv2.imwrite('test.jpg', cv_img)

            if facial_feature not in face_landmark:
                skip = True
                break

        if skip:
            continue

        found_face = True

        if found_face:
            mask_face(face_landmark, mask_img,face_img,pic_path)
        else:
            print("No face is found")


def mask_face(face_landmark, mask_img,face_img,pic_path):
    nose_bridge = face_landmark['nose_bridge']
    len_nose_bridge = len(nose_bridge)
    nose_point = nose_bridge[len_nose_bridge // 2 -1]
    # print(nose_point)

    chin = face_landmark['chin']
    len_chin = len(chin)
    chin_bottom_point = chin[len_chin // 2]
    # print(chin_bottom_point)
    chin_left_point = chin[len_chin // 8]
    chin_right_point = chin[len_chin * 7 // 8]

    width = mask_img.width
    height = mask_img.height
    width_ratio = 1.2
    new_height = int(np.abs(nose_point[1] - chin_bottom_point[1]))
    # print(new_height)

    # left
    # pil image => left,top,right,bottom
    mask_left_img = mask_img.crop((0, 0, width//2, height))
    mask_left_width = get_distance_from_point_to_line(chin_left_point,nose_point,chin_bottom_point)
    mask_left_width = int(mask_left_width * width_ratio)
    mask_left_img = mask_left_img.resize((mask_left_width,new_height))


    # right 
    mask_right_img = mask_img.crop((width//2,0,width,height))
    mask_right_width = get_distance_from_point_to_line(chin_right_point,nose_point,chin_bottom_point)
    mask_right_width = int(mask_right_width * width_ratio)
    mask_right_img = mask_right_img.resize((mask_right_width,new_height))

    # merge mask
    size = (mask_left_img.width + mask_right_img.width, new_height)
    mask_img = Image.new('RGBA',size)
    mask_img.paste(mask_left_img,(0,0),mask_left_img)#,mask_left_img
    mask_img.paste(mask_right_img,(mask_left_img.width,0),mask_right_img)


    # rotate_mask
    angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
    rotated_mask_img = mask_img.rotate(angle,expand=True)
    # rotated_mask_img.show()paste

    # print(rotated_mask_img.width, rotated_mask_img.height)

    # calculate the mask location
    center_x = (nose_point[0] + chin_bottom_point[0]) // 2
    center_y = (nose_point[1] + chin_bottom_point[1]) // 2

    # print(mask_img.width,mask_left_img.width)

    offset = mask_img.width // 2 - mask_left_img.width
    radian = angle * np.pi / 180
    box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2 
    box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

    face_img.paste(rotated_mask_img,(box_x,box_y),rotated_mask_img)
    # face_img.save('test.png')
    # face_img.show()

    path = pic_path.split(os.path.sep)
    name = path[-2]
    image = path[-1]
    
    face_img.save(f'dataset_with_mask/{name}/{image}')





def get_distance_from_point_to_line(point, line_point1, line_point2):
    distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                      (line_point1[0] - line_point2[0]) * point[1] +
                      (line_point2[0] - line_point1[0]) * line_point1[1] +
                      (line_point1[1] - line_point2[1]) * line_point1[0]) / \
        np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
    return int(distance)


# create_mask(image_path)
# plt.imshow(mpimg.imread('test.jpg'))
# plt.show()


