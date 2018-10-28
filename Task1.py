import cv2
import numpy as np

image = cv2.imread('Task1.png', 0)

N,M = image.shape
sobelx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobely = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

kRows=3
kCols=3

#Declaring Empty Images Padded with zeros
sobelxImage=np.asarray([[0] * M for _ in range(N)],dtype=np.float64)
sobelxImage=np.asarray([[0] * M for _ in range(N)],dtype=np.float64)
sobelyImage=np.asarray([[0] * M for _ in range(N)],dtype=np.float64)
sobelGrad=np.asarray([[0] * M for _ in range(N)],dtype=np.float64)

#performing 2D Convolution
kCenterX = int(kCols / 2);
kCenterY = int(kRows / 2);

for i in range(0, N):            

     for j in range(1, M):        
     
         for m in range(0, kRows):  
             
                     mm = kRows - 1 - m
                     for n in range(0, kCols): 
                         nn = kCols - 1 - n  
                         ii = i + (kCenterY - mm)
                         jj = j + (kCenterX - nn)
                         if (ii >= 0 and ii < N and jj >= 0 and jj < M):
                             sobelxImage[i][j]+= (image[ii][jj] * sobelx[mm][nn])
                             sobelyImage[i][j]+= (image[ii][jj] * sobely[mm][nn])

#Calculating Maximum and Minimum value of pixel                           
max=0.0
min=0.0
max1=0.0
min1=0.0
for k in range(1, N-1):
    for l in range(1, M-1): 
        
        if(max<sobelxImage[k][l]):
            max=sobelxImage[k][l]
        if(min>sobelxImage[k][l]):
            min=sobelxImage[k][l]   
        if(max1<sobelyImage[k][l]):
            max1=sobelyImage[k][l]
        if(min1>sobelyImage[k][l]):
            min1=sobelyImage[k][l]

#Method 1
pos_edge_x=(sobelxImage-min)/(max-min)

#Method 2
#pos_edge_x=abs(sobelxImage)/(max)
#cv2.namedWindow('pos_edge_x_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_x_dir', pos_edge_x)
cv2.waitKey(0)
cv2.destroyAllWindows()

pos_edge_y=(sobelyImage-min1)/(max1-min1)
#pos_edge_y=abs(sobelyImage)/(max)
#cv2.namedWindow('pos_edge_y_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_y_dir', pos_edge_y)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Wrting the obtained images
cv2.imwrite('Sobel_Image_gx.png',sobelxImage)
#cv2.imwrite('Pos_Image_x.png',pos_edge_Image) 
cv2.imwrite('Sobel_Image_gy.png',sobelyImage)
