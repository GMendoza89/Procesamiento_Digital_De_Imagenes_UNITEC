import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import myImageProcessing


img = cv.imread('Lenna.png')

histogram = myImageProcessing.normHistogram3Chanel(img)

plt.plot(range(0,256),histogram[:,0],'b')
plt.plot(range(0,256),histogram[:,1],'g')
plt.plot(range(0,256),histogram[:,2],'r')
plt.show()


