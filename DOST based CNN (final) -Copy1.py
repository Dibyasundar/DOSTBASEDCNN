
# coding: utf-8

# In[ ]:


# %load "DOST based CNN for HSI classification.py"


# In[3]:


import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import random
from random import shuffle
from skimage.transform import rotate
import scipy.ndimage
import scipy as sp
from scipy import fftpack
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import spectral
import numpy as np
import scipy
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Conv3D,MaxPooling3D
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.utils import np_utils
import keras
import sys
import matplotlib.pyplot as plt


# In[ ]:


def loadIndianPinesData():
    data_path = os.path.join(os.getcwd(),'data')
    data = sio.loadmat("../../Datasets/Salinas_corrected.mat")['salinas_corrected']
    labels = sio.loadmat("../../Datasets/Salinas_gt.mat")['salinas_gt']

    return data, labels



# In[ ]:


def classifier(X,y,numComponents = 64, windowSize =7,testRatio = 0.95,dost_ = True,wls_alpha=1.0,wls_lambda=1.0,report=None,index=1):

#     reportpath = "../reports/conv_{}_{}_{}_{}_{}_{}.txt".format(numComponents,windowSize,testRatio,dost_,wls_alpha,wls_lambda)

#     with open(reportpath,'w+') as report:

#         print("""         ------------------Iteration Info : ----------------------- \n
#                                 Shape of X :    {}\n
#                                 Shape of y :    {}\n
#                                 numComponents : {}\n
#                                 windowSize :    {}\n
#                                 testRatio :     {}\n
#                                 use dost :      {}\n
#                                 wls_alpha :     {}\n
#                                 wls_lambda :    {}\n
#         """.format(X.shape,y.shape,numComponents,windowSize,testRatio,dost_,wls_alpha,wls_lambda))
#         print("""         ------------------Iteration Info : ----------------------- \n
#                                 Shape of X :    {}\n
#                                 Shape of y :    {}\n
#                                 numComponents : {}\n
#                                 windowSize :    {}\n
#                                 testRatio :     {}\n
#                                 use dost :      {}\n
#                                 wls_alpha :     {}\n
#                                 wls_lambda :    {}\n
#         """.format(X.shape,y.shape,numComponents,windowSize,testRatio,dost_,wls_alpha,wls_lambda),file = report)


    def splitTrainTestSet(X, y, testRatio):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=103,
                                                            stratify=y)
        return X_train, X_test, y_train, y_test


    def dost_bw(l):
        out = np.zeros(int(2*np.log2(l)))
        l1 = np.arange(np.log2(l)-2,-1,-1)
        l2 = np.arange(0,np.log2(l)-1)
        out[1:int(1+np.log2(l)-1)]=l1
        out[-int(np.log2(l)-1):]=l2
        out = np.exp2(out).astype(np.int16)
        return out

    def dost(inp):
        l = inp.shape[0]
        fft_inp = fftpack.fftshift(fftpack.fft(fftpack.ifftshift(inp,axes=0),axis=0),axes=0)
        bw_inp = dost_bw(l)
        k = 0
        dost_inp = np.zeros_like(fft_inp)
        for r in bw_inp:
            if(r==1):
                dost_inp[k,:] = fft_inp[k,:]
                k = k+r
            else:
                dost_inp[k:r+k,:] = fftpack.fftshift(fftpack.ifft(fftpack.ifftshift(fft_inp[k:r+k,:],axes=0),axis=0),axes=0)
                k = k+r
        return dost_inp


    def oversampleWeakClasses(X, y):
        uniqueLabels, labelCounts = np.unique(y, return_counts=True)
        maxCount = np.max(labelCounts)
        labelInverseRatios = maxCount / labelCounts
        newX = X[y == uniqueLabels[0], :, :, :].repeat(round(labelInverseRatios[0]), axis=0)
        newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
        for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
            cX = X[y== label,:,:,:].repeat(round(labelInverseRatio), axis=0)
            cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
            newX = np.concatenate((newX, cX))
            newY = np.concatenate((newY, cY))
        np.random.seed(seed=42)
        rand_perm = np.random.permutation(newY.shape[0])
        newX = newX[rand_perm, :, :, :]
        newY = newY[rand_perm]
        return newX, newY


    def standartizeData(X):
        newX = np.reshape(X, (-1, X.shape[2]))
        scaler = preprocessing.StandardScaler().fit(newX)
        newX = scaler.transform(newX)
        newX = np.reshape(newX, (X.shape[0],X.shape[1],X.shape[2]))
        return newX, scaler

    def applyPCA(X, numComponents=75):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
        return newX, pca

    def padWithZeros(X, margin=2):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX

    def createPatches(X, y, windowSize=5, removeZeroLabels = True):
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = padWithZeros(X, margin=margin)
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r-margin, c-margin]
                patchIndex = patchIndex + 1
        if removeZeroLabels:
            patchesData = patchesData[patchesLabels>0,:,:,:]
            patchesLabels = patchesLabels[patchesLabels>0]
            patchesLabels -= 1
        return patchesData, patchesLabels


    def AugmentData(X_train):
        for i in range(int(X_train.shape[0]/2)):
            patch = X_train[i,:,:,:]
            num = random.randint(0,2)
            if (num == 0):

                flipped_patch = np.flipud(patch)
            if (num == 1):

                flipped_patch = np.fliplr(patch)
            if (num == 2):

                no = random.randrange(-180,180,30)
                flipped_patch = scipy.ndimage.interpolation.rotate(patch, no,axes=(1, 0),
                                                                   reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=False)


        patch2 = flipped_patch
        X_train[i,:,:,:] = patch2

        return X_train

    #    print("Shape of X : ",end='')
    #    print(X.shape)

    if dost_:
        X_dost = np.zeros([X.shape[0],X.shape[1],256])
        X_dost[:,:,:204] = X

        X_ = np.abs(dost(X_dost.reshape([-1,X_dost.shape[2]],order='F').T).T.reshape(X_dost.shape,order='F'))
    else:
        X_ = X

    data_shape = X_.shape
    class_cnt = np.max(y)
    X_pca,pca = applyPCA(X_,numComponents=numComponents)
    XPatches, yPatches = createPatches(X_pca, y, windowSize=windowSize)
    X_train, X_test, y_train, y_test = splitTrainTestSet(XPatches, yPatches, testRatio)
    X_test,X_val, y_test, y_val = splitTrainTestSet(X_test, y_test, 0.3)
    if index ==1:
        print("X_train before over sampling : {}".format(X_train.shape),file = report)
        print("X_train before over sampling : {}".format(X_train.shape))
        print("y_train before over sampling : {}".format(y_train.shape),file = report)
        print("y_train before over sampling : {}".format(y_train.shape))
    X_train, y_train = oversampleWeakClasses(X_train, y_train)
    X_train = AugmentData(X_train)

    for iters in range(1):

        if index==1:
            print("X_train : {}".format(X_train.shape),file = report)
            print("y_train : {}".format(y_train.shape),file = report)
            print("X_test : {}".format(X_test.shape),file = report)
            print("y_test : {}".format(X_test.shape),file = report)
            print("X_val : {}".format(X_val.shape),file = report)
            print("y_val : {}".format(y_val.shape),file = report)

        y_train = np_utils.to_categorical(y_train)
        y_val = np_utils.to_categorical(y_val)
        input_shape = X_train[0].shape
        #    print("Input Shape : {}".format(X_train[...][0].shape))

        C1 = 64

        # Define the model
        model = Sequential()
        model.add(Conv2D(2*C1, (3, 3),data_format="channels_last", activation='relu', input_shape=input_shape))
        model.add(Dropout(0.4))
        # model.add(MaxPooling3D(pool_size=()))
        model.add(Conv2D(C1, (3, 3), data_format="channels_last",activation='relu'))
        model.add(Dropout(0.4))
        # model.add(MaxPooling3D((2,2,2)))
        # model.add(Conv3D(C1, (2,3, 3), activation='relu'))
        # model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(class_cnt, activation='softmax'))

        #     print(model.summary())
        #        print(model.summary(),file = report)

        sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        batch_size = min(1024,X_train.shape[0]//25)
        batch_size  = 64

        filepath = "../models/conv_pavia_{}_{}_{}_{}_{}_{}_{}.h5".format(numComponents,windowSize,testRatio,dost_,wls_alpha,wls_lambda,index)
        #     print(filepath)
        history = model.fit(X_train,
            y_train,
            batch_size=batch_size,
            epochs=500,
            #show_accuracy=False,
            verbose=1,
            validation_data = (X_val,y_val),
            callbacks = [
                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                 keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto'),
        #         keras.callbacks.RemoteMonitor(root='http://localhost:8888')
        #                 keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
            ]
            )


        model.save(filepath=filepath)

        def reports (X_test,y_test,model):
            Y_pred = model.predict(X_test,batch_size=1024)
            y_pred = np.argmax(Y_pred, axis=1)
            target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
                       ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                       'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                       'Stone-Steel-Towers']


            classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
            confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
            score = model.evaluate(X_test, y_test, batch_size=8)
            Test_Loss =  score[0]*100
            Test_accuracy = score[1]*100

            return classification, confusion, Test_Loss, Test_accuracy



        #    y_test_ = np_utils.to_categorical(y_test)
        #
        #    classification, confusion, Test_loss, Test_accuracy = reports(X_test,y_test_,model)
        #    classification = str(classification)
        #    confusion = str(confusion)
        #    numComponents = 64
        #    file_name = 'report' + "WindowSize" + str(windowSize) + "PCA" + str(numComponents) + "testRatio" + str(testRatio) +".txt"
        #    with open(file_name, 'w+') as x_file:
        #        x_file.write('{} Test loss (%)'.format(Test_loss))
        #        x_file.write('\n')
        #        x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
        #        x_file.write('\n')
        #        x_file.write('\n')
        #        x_file.write('{}'.format(classification))
        #        x_file.write('\n')
        #        x_file.write('{}'.format(confusion))

        XPatches, yPatches = createPatches(X_pca, y, windowSize=windowSize,removeZeroLabels=False)
        predictions = model.predict_classes(XPatches,batch_size=1024)+1
        pred_proba = model.predict_proba(XPatches,batch_size=1024)
        predictions[yPatches==0] = 0

        # correct_pred_mask = (predictions[yPatches!=0] == yPatches[yPatches!=0])
        # prob_of_corr_predictions = pred_proba[yPatches!=0,:][correct_pred_mask,:]
        # correct_pred = predictions[yPatches!=0][correct_pred_mask]
        #probs = np.max(pred_proba,axis=1)
        #newXTrain = XPatches[yPatches!=0,...][probs>0.95]
        #newyTrain = yPatches[yPatches!=0][probs>0.95]-1
        #restX = XPatches[yPatches!=0,...][probs<=0.95]
        #resty = yPatches[yPatches!=0,...][probs<=0.95]-1
        #if iters==0:
         #   newTrainX = np.vstack((newXTrain,X_train))
          #  newTrainy = np.vstack((newyTrain[...,np.newaxis],np.argmax(y_train,axis=1)[...,np.newaxis]))[:,0]

        #perm = np.random.permutation(newTrainX.shape[0])

        #newTrainX = newTrainX[perm]
        #newTrainy = newTrainy[perm]

        #X_test,X_val, y_test, y_val = splitTrainTestSet(restX, resty, 0.3)
        #if index ==1:
         #   print("X_train before over sampling : {}".format(newXTrain.shape),file = report)
          #  print("y_train before over sampling : {}".format(newyTrain.shape),file = report)
       # X_train, y_train = oversampleWeakClasses(newTrainX, newTrainy)
    #     X_train = AugmentData(X_train)

        #print("X_train : {}".format(X_train.shape),file = report)
        #print("y_train : {}".format(np_utils.to_categorical(y_train).shape),file = report)
        #print("X_test : {}".format(X_test.shape),file = report)
        #print("y_test : {}".format(X_test.shape),file = report)
        #print("X_val : {}".format(X_val.shape),file = report)
        #print("y_val : {}".format(np_utils.to_categorical(y_val).shape),file = report)

    pred_map = predictions.reshape(data_shape[:2])
    pred_proba = pred_proba.reshape((data_shape[0],data_shape[1],class_cnt))
    def wls_filter(img,lambda_,alpha,L):

        epsilon = 1e-4
        (r,c)=img.shape
        k = r*c

        dy = np.diff(L,1,0)
        dy = -lambda_/(np.power(np.abs(dy),alpha)+epsilon)
        dy = np.lib.pad(dy,[(0,1),(0,0)],'constant',constant_values=0)
        dy = dy.flatten('F')

        dx = np.diff(L,1,1)
        dx = -lambda_/(np.power(np.abs(dx),alpha)+epsilon)
        dx = np.lib.pad(dx,[(0,0),(0,1)],'constant',constant_values=0)
        dx = dx.flatten('F')

        B = np.array([dx,dy])
        A = sp.sparse.diags(B,[-r,-1],(k,k))

        e = dx
        w = np.pad(dx,[(r,0)],'constant',constant_values=0)[:-r]
        s = dy
        n = np.pad(dy,[(1,0)],'constant',constant_values=0)[:-1]

        D = 1-(e+w+s+n)

        A = A + A.T + sp.sparse.diags(D.reshape(1,-1),[0],(k,k))

        result = sp.sparse.linalg.spsolve(A,img.reshape((-1,1),order = 'F'))
        result = result.reshape((r,c),order='F')

        return result

    def bt_wls(Y,prediction,errMat,lamda_,alpha):

        (m,n)=prediction.shape
        bands = Y.shape[2]
    #     Y = Y.reshape((m,n,bands),order = 'F')

        numClasses = errMat.shape[2]
    #     errCube = errMat.reshape((m,n,numClasses),order = 'F')
        errCube = errMat[...]
        errCube = (errCube-np.min(errCube))/(np.max(errCube)-np.min(errCube))

        guidanceImage = applyPCA(Y,1)[0].reshape((m,n))

        #guidanceImage = np.mean(np.log(Y.astype(np.double)+0.0001),2)
        for i in range(numClasses):
            slc = errCube[:,:,i]
            slc[np.logical_not(prediction==i+1)]=0
            slc = wls_filter(slc,lamda_,alpha,guidanceImage)
            errCube[:,:,i]=slc

        new_prediction = np.argmax(errCube,2)+1
        new_prediction[prediction==0] = 0
        return new_prediction


    smoothed_output = bt_wls(X,pred_map,pred_proba,wls_lambda,wls_alpha)

    return (np.sum(pred_map[y!=0]==y[y!=0])/yPatches[yPatches!=0].shape[0],np.sum(smoothed_output[y!=0]==y[y!=0])/yPatches[yPatches!=0].shape[0])

if __name__ == "__main__":
    X,y = loadIndianPinesData()
    wa,wl = 1.2,0.8
    for nc in [32]:
        for ws in [9]:
            for tr in [0.995]:
                for d_ in [True,False]:
                    wof,wf = 0,0
                    reportpath = "../reports/conv_pavia_{}_{}_{}_{}_{}_{}.txt".format(nc,ws,tr,d_,wa,wl)
                    with open(reportpath,'w+') as report:
                        print("""         ------------------Iteration Info : ----------------------- \n
                                    Shape of X :    {}\n
                                    Shape of y :    {}\n
                                    numComponents : {}\n
                                    windowSize :    {}\n
                                    testRatio :     {}\n
                                    use dost :      {}\n
                                    wls_alpha :     {}\n
                                    wls_lambda :    {}\n
                            """.format(X.shape,y.shape,nc,ws,tr,d_,wa,wl),file = report)
                        for i in range(1,2):
                            print('#',end='')
                            t1,t2 = classifier(X,y,numComponents = nc, windowSize =ws,testRatio = tr,dost_ = d_,wls_alpha=wa,wls_lambda=wl,report=report,index=i)
                            wof+=t1
                            wf+= t2
                            print(t1,t2)
                        wof/= 1
                        wf/=1
                        print('\t',end='')
                        print("wof: {}    wf: {}".format(wof,wf),end = '\t')
                        print("Accurcay without filter = {}".format(wof),file=report)
                        print("Accurcay with filter = {}".format(wf),file=report)



