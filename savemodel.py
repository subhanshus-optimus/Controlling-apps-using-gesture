#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 10:36:06 2019

@author: pushap
"""

import tensorflow as tf
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os

from keras.utils import np_utils

currentDir = os.getcwd()

weightFiles = os.listdir("./gesture_weightFilecat3/")
weightFiles.sort()
weightFilepath = currentDir+"/gesture_weightFilecat3/"+weightFiles[-1]
x_train=np.load('x.npy')
y_train=np.load('y.npy')#load the files for the training

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2], 1)
#x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2], 1)#load the files for the training
input_shape=(x_train.shape[1],x_train.shape[2],1)
y_train = np_utils.to_categorical(y_train)
#y_test =np_utils.to_categorical(y_test)
print(input_shape)
model = Sequential()
model.add(Conv2D(64, kernel_size=(7,7), strides=1, activation=tf.nn.relu,input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation=tf.nn.relu,input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten(data_format=None))
#model.add(BatchNormalization())
model.add(Dense(256, activation=tf.nn.relu))
model.add(BatchNormalization())
model.add(Dense(y_train.shape[1],activation=tf.nn.softmax))
model.load_weights(weightFilepath)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.save('my_model1.h5')