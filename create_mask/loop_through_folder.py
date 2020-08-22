import os 
import sys
from tqdm import tqdm
import random
from mask import create_mask

mask_list = []

default_path = os.path.join(os.getcwd(),"create_mask/mask.png")
white_path = os.path.join(os.getcwd(),"create_mask/white.png")
blue_path = os.path.join(os.getcwd(),"create_mask/blue.png")
# black_path = os.path.join(os.getcwd(),"create_mask/black.png")

mask_list.append(default_path)
mask_list.append(white_path)
mask_list.append(blue_path)
# mask_list.append(black_path)

print(mask_list)

dataset_path = 'dataset'

if not os.path.exists('dataset_with_mask'):
    os.mkdir('dataset_with_mask')

for i in os.listdir('dataset'):
    if not os.path.exists(f'dataset_with_mask/{i}'):
        os.mkdir(f'dataset_with_mask/{i}')


imagePaths = []

for i in os.listdir('dataset'):
    for j in os.listdir(f'dataset/{i}'):
        imagePaths.append(f'dataset/{i}/{j}')


for i in tqdm(imagePaths,total=len(imagePaths)):
    mask_path = random.choice(mask_list)
    create_mask(i,mask_path)