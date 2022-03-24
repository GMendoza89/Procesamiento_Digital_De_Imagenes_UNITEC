from cv2 import Canny
import numpy as np
import cv2 as cv
import math as mt

def normHistogram3Chanel(img):
    dimentions = img.shape
    size = int(dimentions[0]*dimentions[1])
    hist = np.zeros([256,3],np.float32)
    elementR = int(0)
    elementG = int(0)
    elementB = int(0)
    for y in range(0,dimentions[0]):
        for x in range(0,dimentions[1]):
            elementR = img[y,x,2]
            elementG = img[y,x,1]
            elementB = img[y,x,0]
            hist[elementR,2] = hist[elementR,2] + 1
            hist[elementG,1] = hist[elementG,1] + 1
            hist[elementB,0] = hist[elementB,0] + 1
    hist /= size
    return hist
    
def normHistogram(img):
    dimentions = img.shape
    size = img.size
    hist = np.zeros(256,np.float32)
    element = int(0)
    for y in range(0,dimentions[0]):
        for x in range(0,dimentions[1]):
            element = img[y,x]
            hist[element] =hist[element] + 1
    hist /= size
    return hist


def lowPassFilter(threshold,img):
    heigh, width = img.shape 
    mask = np.zeros([heigh,width,2],np.uint8)
    (imHR,imWR) = (int(heigh/2),int(width/2))
    mask[imHR-threshold:imHR+threshold,imWR-threshold:imWR+threshold] = 1
    dFImg = cv.dft(np.float32(img),flags=cv.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dFImg)
    fImgShift = dftShift*mask
    bfImgShif=np.fft.ifftshift(fImgShift)
    imgBack = cv.idft(bfImgShif)
    imgBack = cv.magnitude(imgBack[:,:,0],imgBack[:,:,1])
    return imgBack

def highPassFilter(threshold,img):
    heigh, width = img.shape 
    mask = np.ones([heigh,width,2],np.uint8)
    (imHR,imWR) = (int(heigh/2),int(width/2))
    mask[imHR-threshold:imHR+threshold,imWR-threshold:imWR+threshold] = 0
    dFImg = cv.dft(np.float32(img),flags=cv.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dFImg)
    fImgShift = dftShift*mask
    bfImgShif=np.fft.ifftshift(fImgShift)
    imgBack = cv.idft(bfImgShif)
    imgBack = cv.magnitude(imgBack[:,:,0],imgBack[:,:,1])
    return imgBack

def entropy(img):
    H = normHistogram(img)
    E = 0
    for n in range(0,256):
        if H[n] == 0:
            continue
        E += H[n]*abs(mt.log2(H[n]))
    return E

def entropySegmentation(img):
    p = normHistogram(img)
    H = 0
    hLast = 0
   
    bestK = 0
    #calculamos entropias de los pixeles
    for k in range(0,256):
        hBlack = 0
        hWhite = 0
        #Calculamos entropia de valor en negro
        for i in range(0,k):
            if p[i] != 0:
                hBlack =- p[i]*mt.log2(p[i])
        #calculamos la entropia para el blanco
        for i in range(k,256):
            if p[i] != 0:
                hWhite =- p[i]*mt.log2(p[i])
        hLast = hBlack+hWhite
        if hLast > H:
            H = hLast
            bestK = k
    return H,bestK
def segmentation(img,imgSeg, K):
    dimentions = img.shape
    for y in range(0,dimentions[0]):
        for x in range(0,dimentions[1]):
            if img[y,x] < K:
                imgSeg[y,x] = 0
            else:
                imgSeg[y,x] = 255
    return imgSeg

def edgeCannyDetection(img):
    # Pasamos la imagen por un filtro gaussiano
    imgBlurred = cv.GaussianBlur(img, [3,3], 0)
    # convertimos la imagen de color a escala de grices
    imgGray = cv.cvtColor(imgBlurred,cv.COLOR_BGR2GRAY)
    # obtenemos el gradiente en el eje x
    gradX = cv.Sobel(imgGray,cv.CV_16SC1, 1,0)
    # Obtenes el gradien en el eje Y
    gradY = cv.Sobel(imgGray,cv.CV_16SC1, 0,1)
    # Combertimos en profundidad de 8 bits
    gradX_8 = cv.convertScaleAbs(gradX)
    gradY_8 = cv.convertScaleAbs(gradY)

    src = cv.addWeighted(gradX_8,0.5,gradY_8,0.5,0)
    edgeImg = cv.Canny(src,50,100)
    cv.imshow("Canny Edge One",edgeImg)
    edgeImg2 = cv.Canny(gradX,gradY,10,100)
    cv.imshow("Canny Edge Two",edgeImg2)
    edgeImg3 = cv.bitwise_and(img,img,mask=edgeImg2)
    cv.imshow('Bitwise mask',edgeImg3) 

