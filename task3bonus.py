# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:03:44 2018

@author: DELL
"""
import cv2 
import numpy as np

template = cv2.imread('./task3/template.png', 0)
template1 = cv2.imread('./task3_bonus/template1.png', 0)
template2 = cv2.imread('./task3_bonus/template2.png', 0)

image_rbg = cv2.imread('./task3_bonus/t3_6.jpg')
image_grey=cv2.cvtColor(image_rbg, cv2.COLOR_BGR2GRAY)

def Gaussian_Blur(img):
    blur = cv2.GaussianBlur(img,(3,3),0)
    return blur


def laplacian_transf(img):
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    return laplacian

imag=(Gaussian_Blur(image_grey))
img=(laplacian_transf(imag))#transformed image
img1=(laplacian_transf(template))#transformed template
img2=(laplacian_transf(template1))#transformed template
img3=(laplacian_transf(template2))#transformed template

x=np.asarray(img,dtype=np.float32)
y=np.asarray(img1,dtype=np.float32)
z=np.asarray(img2,dtype=np.float32)
a=np.asarray(img2,dtype=np.float32)

w, h = y.shape[::-1]
w1, h1 = z.shape[::-1]
w2, h2 = z.shape[::-1]

method=cv2.TM_CCOEFF_NORMED
res = cv2.matchTemplate(x,y,method)#Template Matching
res1 = cv2.matchTemplate(x,z,method)#Template Matching
res2 = cv2.matchTemplate(x,a,method)#Template Matching
#Template Matching Source Code:OpenCV Site:https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
threshold = .43#thresholding
loc = np.where(res>= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(image_rbg, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

threshold = .42#thresholding
loc = np.where(res1>= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(image_rbg, pt, (pt[0] + w1, pt[1] + h1), (0,255,255), 2)
    
threshold = .42#thresholding
loc = np.where(res2>= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(image_rbg, pt, (pt[0] + w2, pt[1] + h2), (0,255,255), 2)

cv2.imshow('Detected Image',image_rbg)
cv2.waitKey(0)