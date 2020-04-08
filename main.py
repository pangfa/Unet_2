from model import *
from data import *
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import pickle
import keras
from sklearn.utils import shuffle

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


aug_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='nearest')
train_gene = trainGenerator(batch_size=1,aug_dict=aug_args,train_path=r'C:\Users\sen\Desktop\data1\membrane\train',image_folder='image',mask_folder='mask',image_color_mode='rgb',mask_color_mode='rgb',
                            image_save_prefix='image',mask_save_prefix='mask',flag_multi_class=True,save_to_dir = None)
#tensorboard = TensorBoard(log_dir='.\log')
model = unet(num_class=5)
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(train_gene,steps_per_epoch=58,epochs=1,callbacks=[model_checkpoint])

testGene = testGenerator(r'C:\Users\sen\Desktop\data1\membrane\test')
results = model.predict_generator(testGene,6,verbose=1)
saveResult(r'C:\Users\sen\Desktop\data1\membrane\test',results)
