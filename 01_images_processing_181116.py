# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 08:13:20 2018

@author: Veronica Stephens
"""
#%%
#import libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

#%%
def show_image(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

#%%

#DIGITS

#%%
#get list of all image files
import glob, os
path ='C:/Downloads/digits_dataset/digits/'
os.chdir(path)
fileList = []
for file in glob.glob("*.png"):
    fileList.append(path + file)
#fileList

#%%
##ONLY RUN THIS ONCE, IT'S CHANGING FILE NAMES TO MAINTAIN FILE ORDER NECESSARY FOR LABEL CREATION
##replace 010.png w/ 099.png to correct order
#from os import rename
#
#for fname in fileList:
#    if fname.endswith('.png'):
#        new_name = fname.replace('_010', '_099')
#        rename(fname, new_name)

#%%
#import original number files
images = []
for file in fileList:
  img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
  images.append(img)

show_image(images[1])

#%%
#resize all images to 28x28
imagesSmall = []
for x in range(0,len(images)):
#  x=19
  resized_image = cv2.resize(images[x], (28, 28))
#  show_image(resized_image)
  imagesSmall.append(resized_image)

imagesSmall = np.asarray(imagesSmall)
show_image(imagesSmall[1])

#%%
#export images
path_exp ='C:/Downloads/digits_dataset/digits/processed/'

for i in range (0,len(imagesSmall)):
#  i = 1
  file_name = 'img_' + str(i)
  cv2.imwrite(path_exp + file_name + '.png',imagesSmall[i]) #(file name,image you want to save)

#%%
#create corresponding labels for images
import itertools

reps = int(len(imagesSmall)/9)
labels = [1,2,3,4,5,6,7,8,9]
labelsNew = []
for i in range(0,reps):
#  i = 1
  labelsNew.append(labels)

labelsNew  = np.asarray(list(itertools.chain(*labelsNew)))

#%%
#check
i=17
show_image(imagesSmall[i])
labelsNew[i]

#%%
#save image and label arrays
path_exp ='C:/Downloads/digits_dataset/digits/processed/'

np.save(path_exp+'images250.npy', imagesSmall)
np.save(path_exp+'labels250.npy', labelsNew)

#%%

#BORDERS

#%%
#get list of all image files
import glob, os
path ='C:/Downloads/digits_dataset/borders/'
os.chdir(path)
fileList = []
for file in glob.glob("*.png"):
    fileList.append(path + file)
#fileList

#%%
#import original number files
images = []
for file in fileList:
  img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
  images.append(img)

show_image(images[1])

#%%
#resize all images to 28x28
imagesSmall = []
for x in range(0,len(images)):
#  x=19
  resized_image = cv2.resize(images[x], (28, 28))
#  show_image(resized_image)
  imagesSmall.append(resized_image)

imagesSmall = np.asarray(imagesSmall)
show_image(imagesSmall[1])

#%%
#export images
path_exp ='C:/Downloads/digits_dataset/borders/processed/'

for i in range (0,len(imagesSmall)):
#  i = 1
  file_name = 'img_' + str(i)
  cv2.imwrite(path_exp + file_name + '.png',imagesSmall[i]) #(file name,image you want to save)

#%%
#check
i=2
show_image(imagesSmall[i])

#%%
#save image and label arrays
path_exp ='C:/Downloads/digits_dataset/borders/processed/'

np.save(path_exp+'borders.npy', imagesSmall)

