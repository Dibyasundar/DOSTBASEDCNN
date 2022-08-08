#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:20:45 2018

@author: dkoder
"""

import numpy as np
import scipy as sp
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from scipy import io
#%%
f = 1/(20*256*256)  # Frequency, in cycles per second, or Hertz
f_s = 256  # Sampling rate, or number of measurements per second

t = np.linspace(256*16,0,  f_s, endpoint=False)
x = np.sin(f * 2 * np.pi * np.square(t))

fig, ax = plt.subplots()
ax.plot(x)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Signal amplitude');
#%%

X = fftpack.fft(x)/np.sqrt(len((x)))
freqs = fftpack.fftfreq(len(x)) * f_s

fig, ax = plt.subplots()

ax.stem(freqs, np.abs(X))
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-f_s/2,f_s/2 )
ax.set_ylim(-5, 5)

#%%
fig, ax = plt.subplots()
phase = np.angle(X)
ax.stem (freqs,phase)
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-128,128)
ax.set_ylim(-5, 5)
#%%
ffted = fftpack.fft(x)
#%%
x1 = np.arange(np.log2(256) - 2,0, - 1)
x2 = np.arange(0,np.log2(256) - 2)
out =[0]
out.extend(list(x1))
out.extend([0])
out.extend( list(x2))

out = np.array(np.exp2(out),dtype = np.int16)

#%%
k = 1;
res = np.zeros_like(ffted)
for i in out:
    print(i)
    if i == 1:
        k = k + i
    else:
        res[k : k + i - 1] =fftpack.ifft(ffted[k : k + i - 1])
        k = k + i;
        
#%%
ns = 2**10
t = np.linspace(0,1,ns)
f = np.linspace(-ns/2,ns/2 -1 ,ns)
inp = np.exp(2 *np.pi* 1j*(ns/2**2)*t)*((0/8<=t)&(t<1/8))+ \
     np.exp(2 *np.pi* 1j*(ns/2**3)*t)*((1/8<=t)&(t<2/8))+ \
     np.exp(2 *np.pi* 1j*(ns/2**4)*t)*((2/8<=t)&(t<3/8))+ \
     np.exp(2 *np.pi* 1j*(ns/2**5)*t)*((3/8<=t)&(t<4/8))+ \
     np.exp(2 *np.pi* 1j*(ns/2**6)*t)*((4/8<=t)&(t<5/8))+ \
     np.exp(2 *np.pi* 1j*(-ns/2**2)*t)*((5/8<=t)&(t<6/8))+ \
     np.exp(2 *np.pi* 1j*(-ns/2**3)*t)*((6/8<=t)&(t<7/8))+ \
     np.exp(2 *np.pi* 1j*(-ns/2**4)*t)*((7/8<=t)&(t<8/8))
    
#%%

        
#%%
def dostbw(l):
    x1 = np.arange(np.log2(l) - 2,-1, - 1)
    x2 = np.arange(0,np.log2(l) - 1)
    out =[0]
    out.extend(list(x1))
    out.extend([0])
    out.extend( list(x2))
    
    out = np.array(np.exp2(out),dtype = np.int16)     
    return out
        
        
        
#%%
def dost(inp):
    ffted =fftpack.fftshift( fftpack.fft(inp))/np.sqrt(len(inp))
    out = dostbw(len(inp))
    k = 0;
    res = np.zeros_like(ffted)
    for i in out:
#        print(i,k)
        if i == 1:
            res[k] = ffted[k]
            k = k + i
        else:
            res[k : k + i ] =fftpack.ifft(ffted[k : k + i ])
            k = k + i
    res2 = rearrangedost(res)
    plt.matshow(np.abs(res2))
    return res,res2,ffted
        
#%%
def rearrangedost(inp):
    D = len(inp);
    bw = dostbw(D);
    tbw = D / bw;
    out = np.zeros((D, D));
    count = 0
    for hh in np.arange(len(bw)):
        for kk in np.arange(bw[hh]):
#            print("hii")
            ii = D  - np.sum(bw[:hh+1])
            jj = int(tbw[hh] * (kk)) 
            tmp = np.repeat(np.repeat(np.array([[count]]),bw[hh],axis=0),tbw[hh],axis=1).astype(np.int16)
            out[ii:(ii + tmp.shape[0]), jj:(jj + tmp.shape[1])] = tmp;
            count = count + 1;
    
    out = out.astype(np.int32)
    out = inp[out]
    return out
        
#%% 
import cv2
#%%
img = cv2.imread("./codes/ST/lena512.bmp",0)
#%%       
cv2.imshow('Lena',np.log((np.abs(ans))+1))
cv2.waitKey(0)        
        
#%%
def fourier2(inp):
    return fftpack.fftshift(fftpack.fft(fftpack.ifftshift(inp,axes=0),axis=0),axes=0)/np.sqrt(inp.shape[0])
def dost2(inp):     
    ffted =fourier2(inp)
    out = dostbw(inp.shape[0])
    k = 1;
    res = np.zeros_like(ffted)
    for i in out:
        print(i)
        if i == 1:
            k = k + i
        else:
            res[k : k + i - 1,:] =fftpack.ifft(ffted[k : k + i - 1,:],axis=0)
            k = k + i
    return res
def dost2d(inp):
    res = dost2(dost2(inp).T).T
    return res

#%%
ans = dost2d(img)       
        
#%%
#lenafft = fftpack.fftshift(fftpack.fft2(img))
cv2.imshow('Lena',imgr)
cv2.waitKey(0)    
 #%%
modulator_frequency = 4.0
carrier_frequency = 32
modulation_index = 1.0

time = np.arange(4096.0) / 4096.0
modulator = np.sin(2.0 * np.pi * modulator_frequency * time) * modulation_index
carrier = np.sin(2.0 * np.pi * carrier_frequency * time)
product = np.zeros_like(modulator)

for i, t in enumerate(time):
    product[i] = np.sin(2. * np.pi * (carrier_frequency * t + modulator[i]))

plt.subplot(3, 1, 1)
plt.title('Frequency Modulation')
plt.plot(modulator)
plt.ylabel('Amplitude')
plt.xlabel('Modulator signal')
plt.subplot(3, 1, 2)
plt.plot(carrier)
plt.ylabel('Amplitude')
plt.xlabel('Carrier signal')
plt.subplot(3, 1, 3)
plt.plot(product)
plt.ylabel('Amplitude')
plt.xlabel('Output signal')
plt.show()       
#%%
    
matches = []
for i in np.arange(1,17):
    matches.append(np.nonzero(groundT[:,1]==i)[0][0])
    
    
matches = np.array(matches)
#%%
data = img.reshape((-1,200),order='F')
samples = data[matches,:]   
    
    
#%%

fig =plt.figure(figsize=(25,15))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4,4,i+1)#change 4,4 as per the need
    ax.plot(samples[i,:],label='output')
    ax.legend()
    
            
#%%

res = np.zeros((16,256),dtype = np.complex128)
inp = np.zeros(256)
for i in range(16):
    inp[:200] = samples[i,:]
    res[i,:],_,_ = dost(inp)

fig =plt.figure(figsize=(25,15))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4,4,i+1)#change 4,4 as per the need
    ax.plot(np.log(0.000001+np.abs(res[i,:])),label='output')
#    ax.legend()        
        
#%%
def load_data():
    print("Loading Data...")
    img = sp.io.loadmat("../BTC-Demo/Dataset/IndianPines.mat")
    img = img['indian_pines_corrected']
    groundT = sp.io.loadmat("../BTC-Demo/Dataset/IndianPines_groundT.mat")
    groundT = groundT['groundT']
    print("Data Loaded Successfully")
    return img,groundT


def split_dataset(img,groundT):
    print("Splitting Dataset")
    imgr = img.reshape([-1,img.shape[2]],order='F')
    groundIndexes = groundT[:,0]-1
    groundLabels = groundT[:,1]
    trainIndexes,testIndexes,trainLabels,testLabels = train_test_split \
                (groundIndexes,groundLabels,test_size=0.9,random_state=47)
    trainData = imgr[trainIndexes]
#    trainData = normalize(np.array(trainData,dtype=np.float64),axis=1)
    print("Split Successful")
    return (imgr,trainData,trainLabels,testIndexes,testLabels,\
            trainIndexes,groundIndexes,groundLabels)
    
#%%
img,groundT = load_data()
data,trData,trLabels,testIdx,testLabels,trIdx,grIdx,grLabels = split_dataset(img,groundT)
#%%

transformedTrData = np.zeros((trData.shape[0],256),dtype = np.complex128)
transformedTrData2d = np.zeros((trData.shape[0],256,256),dtype = np.complex128)
inp = np.zeros(256)
for i in range(trData.shape[0]):
    inp[:200] = trData[i,:]
    transformedTrData[i,:],transformedTrData2d[i,:],_ = dost(inp)

#%%
transformedTestData = np.zeros((testLabels.shape[0],256),dtype = np.complex128)
transformedTestData2d = np.zeros((testLabels.shape[0],256,256),dtype = np.complex128)

inp = np.zeros(256)
for i in range(testIdx.shape[0]):
    inp[:200] = data[testIdx[i],:]
    transformedTestData[i,:],transformedTestData2d[i,:],_ = dost(inp)     
##%%        
#np.savez_compressed("./data/DOST/train1d.npz",transformedTrData)
#np.savez_compressed("./data/DOST/test1d.npz",transformedTestData)       
#np.savez_compressed("./data/DOST/trLabels1d.npz",trLabels)
#np.savez_compressed("./data/DOST/testLabels1d.npz",testLabels)