import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import myImageProcessing as mip

source = cv.imread("lenna.jpeg")

cv.imshow("Lenna",source)
mip.edgeCannyDetection(source)
cv.waitKey(0)
cv.destroyAllWindows()