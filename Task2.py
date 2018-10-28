# -*- coding: utf-8 -*-
"""
@author: Yogesh
"""
import cv2
import numpy as np
import math as m

image = cv2.imread('./task2/task2.jpg', 0)
N,M=image.shape

def gaussian_func(x,y,sigma):
    a=(-((x**2 + y**2)/(2*sigma**2)))
    b=(1/(2*(3.14)*(sigma**2)))
    return b* m.exp(a)

def g_kernel(sigma):
    k=[]
    s=0
    for i in range(0,7):
        r=[]
        for j in range(0,7):
            value=gaussian_func(j-3,3-i,sigma)
            s=s + value
            r.append(value)
        k.append(r) 
    k1=np.asarray(k)/s
    return k1 
 
def scaling(image):
    N,M=image.shape
    #scaledImage=np.asarray([[0] * int(M/2) for _ in range(int(N/2))],dtype=np.float64)
    scaledImage=[]
    for i in range(0,int((N-1)/2)):
        row=[]
        for j in range(0,int((M-1)/2)):
            row.append(0)
        scaledImage.append(row)
    scaledImage=np.array(scaledImage,dtype=np.float)
    for i in range(1,N):
        for j in range(1,M):
            if(((i+1)%2==0)and((j+1)%2==0)):
                scaledImage[int(i/2)-1][int(j/2)-1]=image[i][j]
    return scaledImage


def Gaus_Blur(kernel,image):
    b,h=image.shape
    empimg = np.asarray([[0.0 for col in range(h+3)] for row in range(b+3)])
    for i in range(3, b-3):
        for j in range(3, h-3):  
            gx = (kernel[0][0] * image[i-3][j-3]) + (kernel[0][1] * image[i-3][j-2]) + \
             (kernel[0][2] * image[i-3][j-1]) + (kernel[0][3] * image[i-3][j]) + \
             (kernel[0][4] * image[i-3][j+1]) + (kernel[0][5] * image[i-3][j+2]) + \
             (kernel[0][6] * image[i-3][j+3]) + (kernel[1][0] * image[i-3][j-3]) + \
             (kernel[1][1] * image[i-3][j-2]) + \
             (kernel[1][2] * image[i-3][j-1]) + (kernel[1][3] * image[i-3][j]) + \
             (kernel[1][4] * image[i-3][j+1]) + (kernel[1][5] * image[i-3][j+2]) + \
             (kernel[1][6] * image[i-3][j+3]) + \
             (kernel[2][0] * image[i-3][j-3]) + \
             (kernel[2][1] * image[i-3][j-2]) + \
             (kernel[2][2] * image[i-3][j-1]) + (kernel[2][3] * image[i-3][j]) + \
             (kernel[2][4] * image[i-3][j+1]) + (kernel[2][5] * image[i-3][j+2]) + \
             (kernel[2][6] * image[i-3][j+3]) +\
             (kernel[3][0] * image[i-3][j-3]) + \
             (kernel[3][1] * image[i-3][j-2]) + \
             (kernel[3][2] * image[i-3][j-1]) + (kernel[3][3] * image[i-3][j]) + \
             (kernel[3][4] * image[i-3][j+1]) + (kernel[3][5] * image[i-3][j+2]) + \
             (kernel[3][6] * image[i-3][j+3]) + \
             (kernel[4][0] * image[i-3][j-3]) + \
             (kernel[4][1] * image[i-3][j-2]) + \
             (kernel[4][2] * image[i-3][j-1]) + (kernel[4][3] * image[i-3][j]) + \
             (kernel[4][4] * image[i-3][j+1]) + (kernel[4][5] * image[i-3][j+2]) + \
             (kernel[4][6] * image[i-3][j+3]) + \
             (kernel[5][0] * image[i-3][j-3]) + \
             (kernel[5][1] * image[i-3][j-2]) + \
             (kernel[5][2] * image[i-3][j-1]) + (kernel[5][3] * image[i-3][j]) + \
             (kernel[5][4] * image[i-3][j+1]) + (kernel[5][5] * image[i-3][j+2]) + \
             (kernel[5][6] * image[i-3][j+3]) + \
             (kernel[6][0] * image[i-3][j-3]) + \
             (kernel[6][1] * image[i-3][j-2]) + \
             (kernel[6][2] * image[i-3][j-1]) + (kernel[6][3] * image[i-3][j]) + \
             (kernel[6][4] * image[i-3][j+1]) + (kernel[6][5] * image[i-3][j+2]) + \
             (kernel[6][6] * image[i-3][j+3])
            empimg[i-3][j-3] = gx
    return empimg

def octave1():
    oct1sig=[0.70,1,1.414,2,2.82]
    for i in range(5):
        sigma=oct1sig[i]
        kernal=g_kernel(sigma)
        dummy=Gaus_Blur(kernal,image)
        cv2.imwrite('./task2/BlurOctave1_%s.png' %i ,dummy)
octave1()
        
def octave2(image1):
    oct2sigma=[1.41,2,2.82,4,5.65]
    for i in range(5):
        sigma=oct2sigma[i]
        kernal=g_kernel(sigma)
        dummy1=Gaus_Blur(kernal,image1)
        cv2.imwrite('./task2/BlurOctave2_%s.png' %i ,dummy1)
oct2image=scaling(image)
octave2(oct2image)
    
def octave3(image2):
    oct3sigma=[2.82,4,5.65,8,11.31]
    for i in range(5):
        sigma=oct3sigma[i]
        kernal=g_kernel(sigma)
        dummy2=Gaus_Blur(kernal,image2)
        cv2.imwrite('./task2/BlurOctave3_%s.png' %i ,dummy2)
octave3(scaling(scaling(image)))

def octave4(image3):
    oct4sigma=[5.65,8,11.31,16,22.62]
    for i in range(5):
        sigma=oct4sigma[i]
        kernal=g_kernel(sigma)
        dummy3=Gaus_Blur(kernal,image3)
        cv2.imwrite('./task2/BlurOctave4_%s.png' %i ,dummy3)
octave4(scaling(scaling(scaling(image))))       

#DOG calculation Octave 1
oct1_0=cv2.imread('./task2/BlurOctave1_0.png',0)
oct1_1=cv2.imread('./task2/BlurOctave1_1.png',0)
oct1_2=cv2.imread('./task2/BlurOctave1_2.png',0)
oct1_3=cv2.imread('./task2/BlurOctave1_3.png',0)
oct1_4=cv2.imread('./task2/BlurOctave1_4.png',0)

dog1_0=oct1_1-oct1_0
dog1_1=oct1_2-oct1_1
dog1_2=oct1_3-oct1_2
dog1_3=oct1_4-oct1_3

#DOG calculation Octave 2
oct2_0=cv2.imread('./task2/BlurOctave2_0.png',0)
oct2_1=cv2.imread('./task2/BlurOctave2_1.png',0)
oct2_2=cv2.imread('./task2/BlurOctave2_2.png',0)
oct2_3=cv2.imread('./task2/BlurOctave2_3.png',0)
oct2_4=cv2.imread('./task2/BlurOctave2_4.png',0)

dog2_0=oct2_1-oct2_0
dog2_1=oct2_2-oct2_1
dog2_2=oct2_3-oct2_2
dog2_3=oct2_4-oct2_3
    
cv2.imwrite('./task2/DOG2_0.png',dog2_0)
cv2.imwrite('./task2/DOG2_1.png',dog2_1)
cv2.imwrite('./task2/DOG2_2.png',dog2_2)
cv2.imwrite('./task2/DOG2_3.png',dog2_3)

#DOG calculation Octave 3
oct3_0=cv2.imread('./task2/BlurOctave3_0.png',0)
oct3_1=cv2.imread('./task2/BlurOctave3_1.png',0)
oct3_2=cv2.imread('./task2/BlurOctave3_2.png',0)
oct3_3=cv2.imread('./task2/BlurOctave3_3.png',0)
oct3_4=cv2.imread('./task2/BlurOctave3_4.png',0)

dog3_0=oct3_1-oct3_0
dog3_1=oct3_2-oct3_1
dog3_2=oct3_3-oct3_2
dog3_3=oct3_4-oct3_3
    
cv2.imwrite('./task2/DOG3_0.png',dog3_0)
cv2.imwrite('./task2/DOG3_1.png',dog3_1)
cv2.imwrite('./task2/DOG3_2.png',dog3_2)
cv2.imwrite('./task2/DOG3_3.png',dog3_3)

#DOG calculation Octave 4
oct4_0=cv2.imread('./task2/BlurOctave4_0.png',0)
oct4_1=cv2.imread('./task2/BlurOctave4_1.png',0)
oct4_2=cv2.imread('./task2/BlurOctave4_2.png',0)
oct4_3=cv2.imread('./task2/BlurOctave4_3.png',0)
oct4_4=cv2.imread('./task2/BlurOctave4_4.png',0)

dog4_0=oct4_1-oct4_0
dog4_1=oct4_2-oct4_1
dog4_2=oct4_3-oct4_2
dog4_3=oct4_4-oct4_3


def findMaximaMinima(DOG,UDOG,BDOG,scale,maxMinPoints):
    for x in range(1,len(DOG)-1):
        for y in range(1,len(DOG[0])-1):
            if((DOG[x][y]>DOG[x-1][y])and(DOG[x][y]>DOG[x+1][y])and(DOG[x][y]>DOG[x-1][y-1])and(DOG[x][y]>DOG[x][y-1])and(DOG[x][y]>DOG[x+1][y-1])and(DOG[x][y]>DOG[x-1][y+1])and(DOG[x][y]>DOG[x][y+1])and(DOG[x][y]>DOG[x+1][y+1])):
                 if((DOG[x][y]>UDOG[x-1][y])and(DOG[x][y]>UDOG[x+1][y])and(DOG[x][y]>UDOG[x-1][y-1])and(DOG[x][y]>UDOG[x][y-1])and(DOG[x][y]>UDOG[x+1][y-1])and(DOG[x][y]>UDOG[x-1][y+1])and(DOG[x][y]>UDOG[x][y+1])and(DOG[x][y]>UDOG[x+1][y+1])and(DOG[x][y]>UDOG[x][y])):
                      if((DOG[x][y]>BDOG[x-1][y])and(DOG[x][y]>BDOG[x+1][y])and(DOG[x][y]>BDOG[x-1][y-1])and(DOG[x][y]>BDOG[x][y-1])and(DOG[x][y]>BDOG[x+1][y-1])and(DOG[x][y]>BDOG[x-1][y+1])and(DOG[x][y]>BDOG[x][y+1])and(DOG[x][y]>BDOG[x+1][y+1])and(DOG[x][y]>BDOG[x][y])):
                            maxMinPoints.append([scale*x,scale*y]);
                      
    return maxMinPoints;

maxmin=[]
maxmin=findMaximaMinima(dog2_2,dog2_1,dog2_3,2,maxmin)
maxmin=findMaximaMinima(dog2_1,dog2_0,dog2_2,2,maxmin)
maxmin=findMaximaMinima(dog3_2,dog3_1,dog3_3,4,maxmin)
maxmin=findMaximaMinima(dog3_1,dog3_0,dog3_2,4,maxmin)

for i in range (len(maxmin)):
    image[maxmin[i][0]][maxmin[i][1]]=255;

print(len(maxmin));
        
cv2.imwrite('Keypoint.png',image)
cv2.waitKey(0)