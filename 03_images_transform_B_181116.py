# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 19:28:55 2018

@author: Veronica Stephens
"""
#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt

#%%
def show_image(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

#%%
#read in scaled images
path ='C:/Downloads/digits_dataset/digits/processed/'
images = np.load(path + 'images250.npy')
labels = np.load(path + 'labels250.npy')

#show_image(images[168])
#print(labels[168])

#%%
#read in border images
path = 'C:/Downloads/digits_dataset/borders/processed/'
borders = np.load(path + 'borders.npy')

#%%
#function for transformations
def imgTransformB (images,borders):
  imagesNew =[]

  for i in range(0,len(borders)):
    #borders (30 total)
#    i=0
    img2 = borders[i]
#    show_image(img2)

    for x in range(0,len(images)):
      #images (450 total)
#      x=0
      img1 = images[x]
#      show_image(img1)

      comb_img = cv2.addWeighted(img1,0.4,img2,0.4,0)
  #    show_image(comb_img)
      thresh = 127
      im_bw = cv2.threshold(comb_img, thresh, 255, cv2.THRESH_BINARY)[1]
#      show_image(im_bw)

      imagesNew.append(im_bw)

  return imagesNew

#%%
#check
imagesNew = imgTransformB(images,borders)

#%%
#check
borderList = [0,35,43,70,82]

for i in range(0,len(borderList)):
  x = borderList[i]
  show_image(imagesNew[x])

#%%
#export images
path_exp = path ='C:/Downloads/digits_dataset/transform_b/'

for i in range (0,len(imagesNew)):
#  i = 1
  file_name = 'img_' + str(i)
  cv2.imwrite(path_exp + file_name + '.png',imagesNew[i]) #(file name,image you want to save)

#%%
#create labels
import itertools

reps = len(borders)
labelsNew = []
for i in range(0,reps):
#  i = 1
  labelsNew.append(labels)

labelsNew  = np.asarray(list(itertools.chain(*labelsNew)))

#%%
#check
i=56
show_image(imagesNew[i])
labelsNew[i]

#%%
#save image and label arrays
path_exp ='C:/Downloads/digits_dataset/transform_b/'

np.save(path_exp+'imagesTB.npy', imagesNew)
np.save(path_exp+'labelsTB', labelsNew)


















