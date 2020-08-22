import torch
import os
import numpy as np
from fastai.vision import *
from fastai.metrics import error_rate,accuracy
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


torch.device('cuda')
torch.cuda.empty_cache()


dataset_path = 'dataset_with_mask'

np.random.seed(42)

trfm = get_transforms(do_flip=True, flip_vert=True, max_zoom=1.2, max_rotate=20.0, max_lighting=0.4)

data = ImageDataBunch.from_folder(dataset_path,train='.',valid_pct =0.2,num_workers = 4).normalize(imagenet_stats)
# ,ds_tfms = trfm

learn = cnn_learner(data,models.resnet34,metrics = [error_rate,accuracy],callback_fns=ShowGraph)



learn.fit_one_cycle(6, max_lr =[1e-6, 1e-4, 1e-3])
# , max_lr =[1e-6, 1e-4, 1e-3]
learn.save('stage-1') 

# learn.unfreeze()
# # learn.model_path = '.'


# learn.lr_find()

# learn.recorder.plot()

# learn.fit_one_cycle(2,max_lr = slice(3e-5,3e-4))
# learn.save('stage-2')


# learn.unfreeze()
# learn.fit_one_cycle(2,max_lr = slice(3e-5,3e-3))
# learn.save('stage-3')


# learn.unfreeze()
# learn.fit_one_cycle(2,max_lr = slice(3e-5,3e-3))
# learn.save('stage-4')


# learn.unfreeze()
# learn.fit_one_cycle(2,max_lr = slice(3e-5,3e-3))
# learn.save('stage-5')



# learn.freeze()
# learn.fit_one_cycle(10,max_lr = slice(3e-5,3e-4))

# learn.save('stage-2')


interp = ClassificationInterpretation.from_learner(learn)

# interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix()


# img = open_image('test_data/00000036.jpg')
# # (image.jpg is any random image.)
# img.show(figsize=(3, 3))
# learn.predict(img)


learn.export()
