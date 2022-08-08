
# coding: utf-8

# In[10]:


# %load finalized_model.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,cohen_kappa_score
from skimage.transform import rotate
import os
import scipy as sp
import scipy.io as sio
import scipy.ndimage
from scipy import fftpack
import keras
from keras.models import load_model
from keras.utils import np_utils
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten,Conv2D,MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')
import pandas as pd
import sys
import itertools
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import spectral
import time
from operator import truediv
# In[43]:


def loadDataset(xpath,ypath,xkey,ykey):
    data = sio.loadmat(xpath)[xkey]
    labels = sio.loadmat(ypath)[ykey]
    return data, labels



def classifier(X,y,numComponents = 64, windowSize =7,testRatio = 0.95,dost_ = True,wls_alpha=1.0,wls_lambda=1.0,report=None,batch_size=32,epochs=200,index=1,bands=None,dataset="indian_pines",target_names=None,seed = 47,valRatio=0.1):



    def splitTrainTestSet(X, y, testRatio,seed):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=seed,
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
            num = np.random.randint(0,2)
            if (num == 0):

                flipped_patch = np.flipud(patch)
            if (num == 1):

                flipped_patch = np.fliplr(patch)
            if (num == 2):

                no = np.random.randrange(-180,180,30)
                flipped_patch = scipy.ndimage.interpolation.rotate(patch, no,axes=(1, 0),
                                                                   reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=False)


        patch2 = flipped_patch
        X_train[i,:,:,:] = patch2

        return X_train

    if dost_:
        n_bands = int(2**np.ceil(np.log2(bands)))
        X_dost = np.zeros([X.shape[0],X.shape[1],n_bands])
        X_dost[:,:,:bands] = X

        X_ = np.abs(dost(X_dost.reshape([-1,X_dost.shape[2]],order='F').T).T.reshape(X_dost.shape,order='F'))
    else:
        X_ = X

    tic = time.clock()

    data_shape = X_.shape
    class_cnt = np.max(y)
    X_pca,pca = applyPCA(X_,numComponents=numComponents)
    XPatches, yPatches = createPatches(X_pca, y, windowSize=windowSize)
    X_train, X_test, y_train, y_test = splitTrainTestSet(XPatches, yPatches, testRatio,seed=seed)
    X_train,X_val, y_train, y_val = splitTrainTestSet(X_train, y_train, valRatio/(1-testRatio),seed=seed)


    indices = []
    for i in range(class_cnt):
        ix = np.isin(yPatches,i)
        idx = np.where(ix)
        np.random.shuffle(idx[0])
        indices += [idx[0]]

    for cl in range(class_cnt):
        if(np.sum(y_train==cl)<15):
            X_train_ = []
            y_train_ = []
            tr_points = indices[cl][:15]
            X_train_ += [XPatches[tr_points,...]]
            y_train_ += [yPatches[tr_points]]

            X_train_ = np.array(X_train_).reshape((-1,windowSize,windowSize,numComponents))
            y_train_ = np.array(y_train_).reshape(-1)

            X_train = np.vstack((X_train,X_train_))
            y_train = np.hstack((y_train,y_train_))

    perm = np.random.permutation(y_train.shape[0])

    X_train = X_train[perm,...]
    y_train = y_train[perm]


    for i in range(class_cnt):
        print("Samples of class {} = {}".format(i,np.sum(y_train==i)))
        print("Samples of class {} = {}".format(i,np.sum(y_train==i)),file = report)

    print("total number of samples = {}".format(y_train.shape[0]))
    print("total number of samples = {}".format(y_train.shape[0]),file = report)
    print("Train test split successful")



    X_train, y_train = oversampleWeakClasses(X_train, y_train)
    X_train = AugmentData(X_train)


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

    print("Building the model")
    C1 = 64

    model = Sequential()
    model.add(Conv2D(C1, (3, 3),data_format="channels_last", activation='relu',padding="same", input_shape=input_shape))
    model.add(Dropout(0.4))
    model.add(Conv2D(2*C1, (3, 3), data_format="channels_last",activation='relu'))
    model.add(Dropout(0.4))
    model.add(Conv2D(4*C1, (3, 3), data_format="channels_last",activation='relu'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_cnt, activation='softmax'))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    batch_size = min(1024,X_train.shape[0]//25)
    modelpath = "../models/conf_conv_{}_{}_{}_{}_{}_{}_{}_{}.h5".format(dataset,numComponents,windowSize,testRatio,dost_,wls_alpha,wls_lambda,index)

    print("Beginning training")
    tic1 = time.clock()
    history = model.fit(X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data = (X_val,y_val),
        callbacks = [
            keras.callbacks.ModelCheckpoint(modelpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
             keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto'),
        ]
        )
    tic2 = time.clock()
    #model.save(filepath=modelpath)

    print("Training time : {}".format(tic2-tic1))

    model = load_model(modelpath)

    def AA_andEachClassAccuracy(confusion_matrix):
        counter = confusion_matrix.shape[0]
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc

    def reports (X,y,y_pred,target_names,model):

        classification = classification_report(y, y_pred,labels =range(1,len(target_names)+1),output_dict =True,target_names=target_names)
        confusion = confusion_matrix(y, y_pred)
        score = model.evaluate(X, np_utils.to_categorical(y-1), batch_size=1024)
        loss =  score[0]*100
        accuracy = accuracy_score(y,y_pred)
        each_acc, average_acc = AA_andEachClassAccuracy(confusion)
        kappa = cohen_kappa_score(y,y_pred)
        classification = pd.DataFrame.from_dict(classification,orient='index')

        return classification, confusion, accuracy, loss,each_acc,average_acc,kappa


    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.figure(figsize=(10,10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)


        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(title)



    print("Begining Classification")
    tic3 = time.clock()

    XPatches, yPatches = createPatches(X_pca, y, windowSize=windowSize,removeZeroLabels=False)
    tic3_ = time.clock()
    predictions = model.predict_classes(XPatches,batch_size=1024)+1
    tic4 = time.clock()
    print("Testing time = {}".format(tic4-tic3))
    print("Testing time without createPatches = {}".format(tic4-tic3_))
    print('Total Time = {}'.format(tic4-tic))


    pred_map = predictions.reshape(data_shape[:2])
    print("Classification Successful")


    nzm =   yPatches!=0

    print("Generating reports")
    def classification_map(map, groundTruth, dpi, savePath):

        fig = plt.figure(frameon=False)
        fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.add_axes(ax)

        ax.imshow(map, aspect='normal')
        fig.savefig(savePath, dpi = dpi)

        return 0


    fig2 = "../images/conf_conv_{}_{}_{}_{}_{}_no_filter_{}_class_map.png".format(dataset,numComponents,windowSize,testRatio,dost_,index)

    y_map = np.zeros((y.shape[0],y.shape[1], 3))

    for item in range(class_cnt):
        if item == 0:
            y_map[pred_map==(item+1)] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y_map[pred_map==(item+1)] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y_map[pred_map==(item+1)] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y_map[pred_map==(item+1)] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y_map[pred_map==(item+1)] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y_map[pred_map==(item+1)] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y_map[pred_map==(item+1)] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y_map[pred_map==(item+1)] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y_map[pred_map==(item+1)] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y_map[pred_map==(item+1)] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y_map[pred_map==(item+1)] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y_map[pred_map==(item+1)] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y_map[pred_map==(item+1)] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y_map[pred_map==(item+1)] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y_map[pred_map==(item+1)] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y_map[pred_map==(item+1)] = np.array([255, 215, 0]) / 255.

    # print y


    classification_map(y_map, y, 24, fig2)
    # spectral.save_rgb(fig1,smoothed_output,colors=spectral.spy_colors)
    # spectral.save_rgb(fig2,pred_map,colors=spectral.spy_colors)


    (cls_rep,conf_mat,acc,loss,each_acc,average_acc,kappa) = reports(XPatches[nzm],yPatches[nzm],predictions[nzm],target_names,model)

    print("\nscores without filter\n")
    print("Accuracy : {}".format(acc))
    print(cls_rep)

    print(each_acc)
    print(average_acc)
    print(kappa)


    pkl2 = "../excelsheets/conf_conv_{}_{}_{}_{}_{}_no_filter_{}.pkl".format(dataset,numComponents,windowSize,testRatio,dost_,index)
    cls_rep.to_pickle(pkl2)

    fig2 = "../images/conf_conv_{}_{}_{}_{}_{}_no_filter_{}.png".format(dataset,numComponents,windowSize,testRatio,dost_,index)
    plot_confusion_matrix(conf_mat,classes=target_names,normalize=True,title = fig2)

    print("Returning Accuracies")

    return (np.sum(pred_map[y!=0]==y[y!=0])/yPatches[yPatches!=0].shape[0])


# In[44]:


if __name__ == "__main__":

    dataset_ = 2
    if dataset_==1:
        xpath = "../../Datasets/Indian_pines_corrected.mat"
        ypath = "../../Datasets/Indian_pines_gt.mat"
        xkey  ="indian_pines_corrected"
        ykey  = "indian_pines_gt"
        bands=200
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
                       ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                       'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                      'Stone-Steel-Towers']
        tr=0.8
        vlr = 0.1

    else:
        xpath = "../../Datasets/PaviaU.mat"
        ypath = "../../Datasets/PaviaU_gt.mat"
        xkey  ="paviaU"
        ykey  = "paviaU_gt"
        bands=103
        target_names = ['Asphalt','Meadows','Gravel','Trees','Painted-metal-sheets','Bare-soil',
                                           'Bitumen','Self-Blocking-bricks','Shadows']
        tr=0.9
        vlr = 0.05


    # target_names = ['Scrub','Willow-swamp','CP-hammock','Slash-pine','Oak/Broadleaf','Hardwood',
    #                      'Swap','Graminoid-marsh','Spartina-marsh','Cattail-marsh',
    #                      'Salt-marsh','Mud-flats','Water']
    # target_names = ["Weeds-1",'Weeds-2','Fallow-plow','Fallow-smooth','Stubble',
    #                  'Celery','Grapes-untrained','Soil','Corn','Soyabean-notill','Lettuce-4wk',
    #                  'Lettuce-5wk','Lettuce-6wk','Lettuce-7wk','Vineyard-untrained','Vineyard-trellis']

    X,y = loadDataset(xpath,ypath,xkey,ykey)
    print("Dataset loaded successfully")
    wof,wf = 0,0
    nc=32
    ws=9
    d_= True
    wa=1.2
    wl=0.8

    reportpath = "../reports/conf_conv_{}_{}_{}_{}_{}_{}_{}.txt".format(xkey,nc,ws,tr,d_,wa,wl)
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
        accuracies = []
        for i in range(5):
            acc = classifier(X,y,numComponents = nc, windowSize =ws,testRatio = tr,dost_ = d_,wls_alpha=wa,wls_lambda=wl,report=report,batch_size=128,epochs=1000,bands=bands,target_names=target_names,dataset=xkey,index=i,seed = np.random.randint(43,1001),valRatio=vlr)
            accuracies += [acc]
        accuracies = np.array(accuracies)
        print('')
        print("wof: {}+-{}  ".format(np.mean(accuracies),np.std(accuracies)))
        print("wof: {}+-{}  ".format(np.mean(accuracies),np.std(accuracies)),file = report)

# In[1]:



