# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 10:20:26 2018

@author: Yogesh
"""
import cv2 
import numpy as np

template = cv2.imread('./task3/template.png', 0)
image_rbg = cv2.imread('./task3/pos_15.jpg')
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

x=np.asarray(img,dtype=np.float32)
y=np.asarray(img1,dtype=np.float32)

w, h = y.shape[::-1]

method=cv2.TM_CCOEFF_NORMED
res = cv2.matchTemplate(x,y,method)#Template Matching

#Template Matching Source Code:OpenCV Site:https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
threshold = .42#thresholding
loc = np.where(res>= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(image_rbg, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('Detected Image',image_rbg)
cv2.waitKey(0)