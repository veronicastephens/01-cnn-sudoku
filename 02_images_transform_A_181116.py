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
#read image npy files
path ='C:/Downloads/digits_dataset/digits/processed/'
images = np.load(path + 'images250.npy')

show_image(images[17])

#%%

#TRANSFORMATIONS: IMAGES


#%%
#function for 10 image transformations
def imgTransformA (img):
  rows,cols = img.shape
  #transformations
  M1 = np.float32([[1,0,2.5],[0,1,2.5]]) #2 sides upper left corner
  M2 = np.float32([[1,0,-2.5],[0,1,-2.5]]) #2 sides lower right corner
  M3 = np.float32([[1,0,2.5],[0,1,0]]) #1 side left side
  M4 = np.float32([[1,0,-2.5],[0,1,0]]) #1 side right side
  M5 = np.float32([[1,0,0],[0,1,-2.5]]) #1 side bottom
  M6 = np.float32([[1,0,0],[0,1,2.5]]) #1 side top
  M7 = np.float32([[1,0.1,0],[0.15,1,-2.5]]) #2 sides bottom, 1 right top skewed
  M8 = np.float32([[1,-0.1,0],[0.1,1,-2.5]]) #2 sides bottom, 1 right top skewed
  M9 = np.float32([[1,-0.1,0],[-0.1,1,-2.5]]) #2 sides bottom right, skewed
  M10 = np.float32([[1,0.1,0],[-0.1,1,-2.5]]) #2 sides bottom left, skewed

  M11 = np.float32([[1,0,3.5],[0,1,1.5]])
  M12 = np.float32([[1,0,-2.5],[0,1,-2.5]])
  M13 = np.float32([[1,0,4.5],[0,1,0]])
  M14 = np.float32([[1,0,-3.5],[0,1,0]])
  M15 = np.float32([[1,0,0],[0,1,-1.5]])
  M16 = np.float32([[1,0,0],[0,1,3.5]])
  M17 = np.float32([[1,0.1,0],[0.15,1,-3.5]])
  M18 = np.float32([[1,-0.1,0],[0.1,1,-2.5]])
  M19 = np.float32([[1,-0.1,0],[-0.1,1,-3.5]])
  M20 = np.float32([[1,0.1,0],[-0.1,1,-1.5]])

  M21 = np.float32([[1,0,2.5],[0,1,.5]])
  M22 = np.float32([[1,0,-2.5],[0,1,-3.5]])
  M23 = np.float32([[1,0,2.5],[0,1,0]])
  M24 = np.float32([[1,0,-3.5],[0,1,0]])
  M25 = np.float32([[1,0,0],[0,1,-3.5]])
  M26 = np.float32([[1,0,-1],[0,1,3.5]])
  M27 = np.float32([[1,0,0],[0,1,-1.5]])
  M28 = np.float32([[1,0,0],[0.1,1,-3.5]])
  M29 = np.float32([[1,0,0],[0,1,-1.5]])
  M30 = np.float32([[1,0,0],[0,1,-2.5]])

  transform =[M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,
              M11,M12,M13,M14,M15,M16,M17,M18,M19,M20,
              M21,M22,M23,M24,M25,M26,M27,M28,M29,M30]
  imagesNew =[]

  for i in transform:
#    i=transform[15]
#    img = imagesSmall[168]
    dst = cv2.warpAffine(img,i,(cols,rows),flags=cv2.INTER_LINEAR) #borderMode=cv2.BORDER_REPLICATE)
    imagesNew.append(dst)
#    show_image(dst)
  return imagesNew

#%%
#get image transformations
imagesSmall = images

imagesTransform = []
for i in range(0,len(imagesSmall)):
    imagesNew = imgTransformA(imagesSmall[i])
    for i in range(0,len(imagesNew)):
      imagesTransform.append(imagesNew[i])

#%%
#plot to check
for i in range(0,31):
  show_image(imagesTransform[i])

#%%
#export images
path_exp = 'C:/Downloads/digits_dataset/transform_a/'

for i in range (0,len(imagesTransform)):
#  i = 1
  file_name = 'img_' + str(i)
  cv2.imwrite(path_exp + file_name + '.png',imagesTransform[i]) #(file name,image you want to save)

#%%

# TRANSFORMATIONS: LABELS


#%%
#create corresponding labels for images
import itertools
#get number of images
reps = int((len(imagesTransform)/len(imagesSmall)))
fonts = int(len(imagesSmall)/9)

labelList = []
for i in range(1,10):
  test = list(itertools.repeat(i,reps))
  labelList.append(test)

flattened_list = []
for x in labelList:
    for y in x:
        flattened_list.append(y)
flattened_list

labels = np.asarray(flattened_list*fonts)
print(labels.shape)
#print(labels)

#%%
#check
i=513
show_image(imagesTransform[i])
labels[i]

#%%
#save image and label arrays
path_exp = 'C:/Downloads/digits_dataset/transform_a/'

np.save(path_exp+'imagesTA.npy', imagesTransform)
np.save(path_exp+'labelsTA.npy', labels)







