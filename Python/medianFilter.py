import cv2 as cv
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import sys
import myImageProcessing as mip

# if len(sys.argv) >= 2:
#     imgFile = sys.argv[1]
# else:
#     #print("Por favor ingresa la ruta o nombre del archivo")
#     imgFile = "Lenna.png"

img = cv.imread("Lenna_Noise.png",0)
imgOriginal = cv.imread("lenna.jpeg",0)
#Filtro de media
imgFM = cv.medianBlur(img,3)

plt.subplot(131)
plt.imshow(img,cmap="gray")
plt.subplot(132)
plt.imshow(imgFM,cmap="gray")
plt.subplot(133)
plt.imshow(imgOriginal,cmap="gray")
plt.show()