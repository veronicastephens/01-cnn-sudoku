# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 08:13:20 2018

@author: Veronica Stephens
"""
#%%
#import libraries
#import cv2
import numpy as np
from matplotlib import pyplot as plt

#%%
def show_image(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

#%%

# COMBINE NPY IMAGES & LABELS


#%%
#read image and label arrays

#transformation a
path ='C:/Downloads/digits_dataset/transform_a/'
images1 = np.load(path + 'imagesTA.npy')
labels1 = np.load(path + 'labelsTA.npy')

#transformation b
path ='C:/Downloads/digits_dataset/transform_b/'
images2 = np.load(path + 'imagesTB.npy')
labels2 = np.load(path + 'labelsTB.npy')

#%%
#combine data
imagesC = np.vstack((images1,images2))
labelsC = np.append(labels1,labels2)

#%%
#check
i=620
show_image(imagesC[i])
labelsC[i]

#%%
#save image and label arrays
path_exp = 'C:/Downloads/digits_dataset/combined/'

np.save(path_exp+'imagesT_AB.npy', imagesC)
np.save(path_exp+'labelsT_AB.npy', labelsC)

#%%
#check labels by saving to csv (create pivot talbe, get same # / digit?)
path_exp = 'C:/Downloads/digits_dataset/combined/'
np.savetxt(path_exp+"testT_AB.csv", labelsC, delimiter=",")







