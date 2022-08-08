#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 16:18:13 2018

@author: dkoder
"""

#%%
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, LSTM, Reshape, Input, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt
from scipy import io
from sklearn.preprocessing import normalize
#%%


def load_data():
    print("Loading Data...")
    img = io.loadmat("../BTC-Demo/Dataset/IndianPines.mat")
    img = img['indian_pines_corrected']
    groundT = io.loadmat("../BTC-Demo/Dataset/IndianPines_groundT.mat")
    groundT = groundT['groundT']
    print("Data Loaded Successfully")
    return img, groundT


def split_dataset(img, groundT):
    print("Splitting Dataset")
    imgr = img.reshape([-1, img.shape[2]], order='F').astype(np.float32)
    imgr = (imgr - np.mean(imgr, axis=0)) / np.var(imgr, axis=0)
    groundIndexes = groundT[:, 0] - 1
    groundLabels = groundT[:, 1]
    trainIndexes, testIndexes, trainLabels, testLabels = train_test_split(
        groundIndexes, groundLabels, test_size=0.8, random_state=47)
    trainData = imgr[trainIndexes]
#    trainData = normalize(np.array(trainData,dtype=np.float64),axis=1)
    print("Split Successful")
    return (imgr, trainData, trainLabels, testIndexes, testLabels,
            trainIndexes, groundIndexes, groundLabels)


#%%
img, groundT = load_data()
data, trData, trLabels, testIdx, testLabels, trIdx, grIdx, grLabels = split_dataset(
    img, groundT)
#%%
trData = np.load("./data/DOST/train1d.npz").items()[0][1]
testData = np.load("./data/DOST/test1d.npz").items()[0][1]
trLabels = np.load("./data/DOST/trLabels1d.npz").items()[0][1]
testLabels = np.load("./data/DOST/testLabels1d.npz").items()[0][1]
#%%
trOnehot = to_categorical(trLabels - 1)
testOnehot = to_categorical(testLabels - 1)

# %%
trData = np.log(0.000001 + np.abs(trData))
#testData = np.abs(testData)
trData = (trData - np.mean(trData, axis=0, keepdims=True)) / \
    (0.0000001 + np.var(trData, axis=0, keepdims=True))
#%%
inpL = Input(batch_shape=(128, 256, 1))

#conv = Conv1D(16,(11,),activation='relu')(inpL)
#conv = MaxPooling1D()(conv)
#
#conv = Conv1D(32,(11,),activation='relu')(conv)
#conv = Dropout(0.5)(conv)
#conv = MaxPooling1D()(conv)
##
#conv = Conv1D(64,(9,),activation='relu')(conv)
#conv = Dropout(0.5)(conv)
#conv = MaxPooling1D()(conv)
#
#conv = Conv1D(128,(7,),activation='relu')(conv)
#conv = Dropout(0.4)(conv)
#conv = MaxPooling1D()(conv)
lstm = LSTM(32, stateful=True, return_sequences=True)(inpL)
lstm = Dropout(0.4)(lstm)
lstm = LSTM(64, stateful=True, return_sequences=False)(lstm)
lstm = Dropout(0.4)(lstm)
#fc = Flatten()(lstm)
fc = Dense(384, activation='relu')(lstm)
fc = Dropout(0.4)(fc)
fc = Dense(128, activation='relu')(fc)
fc = Dropout(0.4)(fc)

fc = Dense(16, activation='softmax')(fc)

model = Model(inputs=(inpL), outputs=(fc))
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

#%%

history = model.fit(x=trData[:, :, np.newaxis], y=trOnehot[:, :],
                    validation_split=0.125, batch_size=128, epochs=100)

#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

#%%
predictions = model.predict(data[grIdx, :, np.newaxis])

#%%

pred = np.zeros((145 * 145))

pred[grIdx] = np.argmax(predictions, axis=1)
pred = pred.reshape((145, 145), order='F')
plt.matshow(pred)
