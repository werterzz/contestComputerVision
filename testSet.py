from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, UpSampling2D

from keras.models import load_model
import cv2
import numpy as np
#edit with your model
IMAGE_SIZE = (256,256)

model = Sequential()
model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                 input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal'))

model.load_weights('my_model.h5')

import glob

path = glob.glob("Test/Input/*.jpg")

for myfile in path:
    test_im = cv2.imread(myfile)
    true_size = test_im.shape
    imshow_size = (512,round(true_size[0]*512/true_size[1]))
    #cv2.imshow('Input',cv2.resize(test_im, imshow_size))

    test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
    test_im = cv2.resize(test_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    test_im = test_im/255.
    test_im = np.expand_dims(test_im, axis=0)
    segmented = model.predict(test_im)
    segmented = np.around(segmented)
    segmented = (segmented[0, :, :, 0]*255).astype('uint8')
    im_pred = cv2.resize(segmented, imshow_size)
    #cv2.imshow('Output',im_pred)
    im_pred = cv2.resize(im_pred, (true_size[1],true_size[0]), interpolation = cv2.INTER_AREA)
    #im_true =  cv2.resize(im_true, IMAGE_SIZE)
    #im_pred =  cv2.resize(im_pred, IMAGE_SIZE)
    myfile = myfile.replace("Input","Output")
    cv2.imwrite(myfile,im_pred)
    #cv2.waitKey()
    print(myfile)