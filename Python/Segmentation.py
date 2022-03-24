import cv2 as cv
import myImageProcessing as mip
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread("rose.jpg",0)


imgSeg = img.copy()

H,K = mip.entropySegmentation(img)
print(H,K)
mip.segmentation(img,imgSeg,K)
########## Graficamos las Imagenen

plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.subplot(122)
plt.imshow(imgSeg,cmap='gray')
plt.show()