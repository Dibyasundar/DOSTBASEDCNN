#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 10:20:17 2018

@author: dkoder
"""

import numpy  as np
from keras.models import Model
from keras.layers import Dense,Dropout,Flatten,Input,Conv2D,MaxPooling2D,ConvLSTM2D
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras import optimizers
from keras import backend as K
#%%
batch_size = 50
epochs = 5
num_classes = 10
#%%
(trainX,trainY),(testX,testY) = cifar10.load_data()
#%%
trainX = trainX.astype(np.float32)
testX = testX.astype(np.float32)
trainY_ohenc= to_categorical(trainY)
testY_ohenc=to_categorical(testY)
#%%
trainX = trainX[:,np.newaxis,:,:,:].repeat(4,axis=1)
testX = testX[:,np.newaxis,:,:,:].repeat(4,axis=1)

#%%

trainX=(trainX)/255
testX = (testX)/255

#%%
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("tf")

#%%

inpl = Input(shape=(None,32,32,3))

rcnn = ConvLSTM2D(32,(3,3),activation='relu',dropout=0.4,return_sequences=True)(inpl)
#rcnn = MaxPooling2D()(rcnn)

rcnn = ConvLSTM2D(32,(3,3),activation='relu',dropout=0.4,return_sequences=True)(rcnn)
#rcnn = MaxPooling2D()(rcnn)


rcnn = ConvLSTM2D(32,(5,5),activation='relu',dropout=0.4,return_sequences=False)(rcnn)
rcnn = MaxPooling2D()(rcnn)

dense = Flatten()(rcnn)

dense = Dense(10,activation='softmax')(dense)

model = Model(inputs=(inpl,),outputs=(dense,))

#sgd = optimizers.SGD(lr=0.1,momentum=0.99,decay=0.0001,nesterov=True)

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

#%%

history = model.fit(x=trainX,y=trainY_ohenc,validation_split=0.2,batch_size=batch_size,epochs=epochs)

#%%
model.predict(testX,batch_size=)
