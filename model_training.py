
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import os
import cv2
import shutil
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"



try:
    shutil.rmtree("gesture_weightFilecat3")
except OSError as e:
#    print ("Error: %s - %s." % (e.filename, e.strerror))
    k=1
    
    
os.mkdir("gesture_weightFilecat3")
currentDir = os.getcwd()


x_train=np.load('x.npy',allow_pickle=True)
y_train=np.load('y.npy',allow_pickle=True)#load the files for the training


#x_test=np.save('x_test.npy',x_test)
#y_test=np.save('y_test.npy',y_test)
print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2], 1)#load the files for the training
print(x_train.shape)
input_shape=(x_train.shape[1],x_train.shape[2],1)
y_train = np_utils.to_categorical(y_train)
#print(y_train[0])
#cv2.imshow(x_train[0][0])
#cv2.waitKey(0)
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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
filepath = currentDir+"/gesture_weightFilecat3/gesturecat3-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(x=x_train,y=y_train, epochs=100, batch_size=50, callbacks= callbacks_list)