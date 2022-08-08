#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:14:36 2018

@author: dkoder
"""

#%%
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Conv1D,MaxPooling1D,Dense,Dropout,Flatten,LSTM,Reshape,Input,concatenate
from keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt
#%%
trData = np.load("./data/DOST/train1d.npz").items()[0][1]
testData = np.load("./data/DOST/test1d.npz").items()[0][1]
trLabels = np.load("./data/DOST/trLabels1d.npz").items()[0][1]-1
testLabels = np.load("./data/DOST/testLabels1d.npz").items()[0][1]-1
#%%
trOnehot = to_categorical(trLabels)
testOnehot = to_categorical(testLabels)

#%%
trData = np.abs(trData)
testData = np.abs(testData)
#%%
inpL = Input(batch_shape=(32,256,1))

conv = Conv1D(16,(5,),activation='relu')(inpL)
conv = MaxPooling1D()(conv)

conv = Conv1D(32,(5,),activation='relu')(conv)
conv = Dropout(0.3)(conv)
conv = MaxPooling1D()(conv)

conv = Conv1D(64,(3,),activation='relu')(conv)
conv = Dropout(0.3)(conv)
conv = MaxPooling1D()(conv)

lstm = LSTM(64,stateful=True,return_sequences=True)(inpL)
lstm = LSTM(64)(lstm)
lstm = Dropout(0.3)(lstm)
conv = Flatten()(conv)

fc = concatenate([conv,lstm])

fc = Dense(512,activation='relu')(fc)
fc = Dropout(0.4)(fc)

fc = Dense(16,activation='softmax')(fc)

model = Model(inputs=(inpL),outputs=(fc))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

#%%

history = model.fit(x=trData[:,:,np.newaxis],y=trOnehot,validation_split=0.125,batch_size=32,epochs=10)
#%%
model.predict(trData[:,:,np.newaxis],batch_size=128)
