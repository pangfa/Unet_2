import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
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


IMAGE_SIZE = 256

def unet(pretrained_weights = None, input_size = (IMAGE_SIZE,IMAGE_SIZE,3),num_class=5):
    inputs = Input(input_size)
    conv1 = Conv2D(64,3,activation=None,padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    conv1 = Conv2D(64,3,activation=None,padding='same',kernel_initializer='he_normal')(conv1)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    pool1 = MaxPool2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128,3,activation=None,padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    conv2 = Conv2D(128,3,activation=None,padding='same',kernel_initializer='he_normal')(conv2)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    pool2 = MaxPool2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(256,3,activation=None,padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    conv3 = Conv2D(256,3,activation=None,padding='same',kernel_initializer='he_normal')(conv3)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    pool3 = MaxPool2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(512,3,activation=None,padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
    conv4 = Conv2D(512,3,activation=None,padding='same',kernel_initializer='he_normal')(conv4)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPool2D(pool_size=(2,2))(drop4)

    conv5 = Conv2D(1024,3,activation=None,padding='same',kernel_initializer='he_normal')(pool4)
    conv5 = LeakyReLU(alpha=0.3)(conv5)
    conv5 = Conv2D(1024,3,activation=None,padding='same',kernel_initializer='he_normal')(conv5)
    conv5 = LeakyReLU(alpha=0.3)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512,2,activation=None,padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop5))
    up6 = LeakyReLU(alpha=0.3)(up6)
    merge6 = concatenate([drop4,up6],axis=3)
    conv6 = Conv2D(512,3,activation=None,padding='same',kernel_initializer='he_normal')(merge6)
    conv6 = LeakyReLU(alpha=0.3)(conv6)
    conv6 = Conv2D(512,3,activation=None,padding='same',kernel_initializer='he_normal')(conv6)
    conv6 = LeakyReLU(alpha=0.3)(conv6)

    up7 = Conv2D(256,3,activation=None,padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
    up7 = LeakyReLU(alpha=0.3)(up7)
    merge7 = concatenate([conv3,up7],axis=3)
    conv7 = Conv2D(256,3,activation=None,padding='same',kernel_initializer='he_normal')(merge7)
    conv7 = LeakyReLU(alpha=0.3)(conv7)
    conv7 = Conv2D(256,3,activation=None,padding='same',kernel_initializer='he_normal')(conv7)
    conv7 = LeakyReLU(alpha=0.3)(conv7)


    up8 = Conv2D(128,3,activation=None,padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    up8 = LeakyReLU(alpha=0.3)(up8)
    merge8 = concatenate([conv2,up8],axis=3)
    conv8 = Conv2D(128,3,activation=None,padding='same',kernel_initializer='he_normal')(merge8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)
    conv8 = Conv2D(128,3,activation=None,padding='same',kernel_initializer='he_normal')(conv8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)

    up9 = Conv2D(64,3,activation=None,padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    up9 = LeakyReLU(alpha=0.3)(up9)
    merge9 = concatenate([conv1,up9],axis=3)
    conv9 = Conv2D(64,3,activation=None,padding='same',kernel_initializer='he_normal')(merge9)
    conv9 = LeakyReLU(alpha=0.3)(conv9)
    conv9 = Conv2D(64,3,activation=None,padding='same',kernel_initializer='he_normal')(conv9)
    conv9 = LeakyReLU(alpha=0.3)(conv9)

    if num_class == 2:#判断是否多分类问题，二分类激活函数使用sigmoid，而多分割采用softmax。相应的损失函数的选择也各有不同。
        conv10 = Conv2D(1,1,activation='sigmoid')(conv9)
        loss_function = 'binary_crossentropy'
    else:
        conv10 = Conv2D(num_class,1,activation='softmax')(conv9)
        loss_function = 'categorical_crossentropy'

    # conv10 = Conv2D(3,1,activation='softamx')(conv9)

    model = Model(input= inputs,output=conv10)

    model.compile(optimizer=Adam(lr = 1e-6),loss=loss_function,metrics=['accuracy'])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
