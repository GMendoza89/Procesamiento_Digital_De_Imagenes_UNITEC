import cv2 as cv                    # Libreria para manejo de imagen
import numpy as np                  # Librería para manejo de matrices
import matplotlib.pyplot as plt

image = cv.imread("lenna.jpeg",1)   # leemos la imagen en un espacio de color
gray = cv.imread("lenna.jpeg",0)    # lectura en escala de grises
grayDimentions = gray.shape         # Obtenemos dimensiones
(h, w) = gray.shape                 # Obtenemos dimensiones en una tupla h - alto w - ancho
pixels = gray.size                  # Obtenemos el total de pixeles de la imagen

imageOutput = np.zeros([h,w,3],np.uint8)    # creamos imagen de salida

hRed = np.zeros((256),np.uint16)    # Creamos vector de datos para intecidades en cada canal
hBlue = np.zeros((256),np.uint16)
hGreen = np.zeros((256),np.uint16)

for y in range(0,h):                # Leemos cada pixel y almacenamos sumamos uno mas en el arreglo
    for x in range(0,w):
        intRed = image[y,x,2]       
        intBlue = image[y,x,0]
        intGreen = image[y,x,1]

        hRed[intRed] +=1
        hBlue[intBlue] +=1
        hGreen[intGreen] +=1

mRed = 0.0                          # generamos valor medio para umbralizar en cada canal
mBlue = 0.0
mGreen = 0.0
for element in range(0,256):       #generamos promedio de intencidad de cada canal
    mRed += float(hRed[element]*element)
    mBlue += float(hBlue[element]*element)
    mGreen += float(hGreen[element]*element)

mRed /= pixels
mBlue /= pixels
mGreen /= pixels

#pltoMedRed = np.ones((256))*mRed
#pltoMedBlue = np.ones((256))*mBlue
#pltoMedGreen = np.ones((256))*mGreen

print("Valor medio de intencidades de rojo es: "+str(mRed))
print("Valor medio de intencidades de Azul es: "+str(mBlue))
print("Valor medio de intencidades de verde es: "+str(mGreen))

plt.plot(hRed,'r')
plt.plot(pltoMedRed,'r')
plt.plot(hBlue,'b')
plt.plot(pltoMedBlue,'b')
plt.plot(hGreen,'g')
plt.plot(pltoMedGreen,'g')
plt.title("Histograma de color")
plt.show()


umbralRed = int(mRed)
umbralGreen = int(mGreen)
umbralBlue = int(mBlue)

imRed = np.zeros((h,w),np.uint8)
imGreen = np.zeros((h,w),np.uint8)
imBlue = np.zeros((h,w),np.uint8)
imZero = np.zeros((h,w),np.uint8)


for y in range(0,h):
    for x in range(0,w):
        if image[y,x,2] > umbralRed:
            imRed[y,x] = 255
        else:
            imRed[y,x] = 0 
        if image[y,x,1] > umbralGreen:
            imGreen[y,x] = 255
        else:
            imGreen[y,x] = 0
        if image[y,x,0] > umbralBlue:
            imBlue[y,x] = 255
        else:
            imBlue[y,x] = 0

imgRed = cv.merge([imRed,imZero,imZero])
imgGreen = cv.merge([imZero,imGreen,imZero])
imgBlue = cv.merge([imZero,imZero,imBlue])
umbIMG = cv.merge([imBlue,imGreen,imRed])

plt.subplot(221)
plt.imshow(imgRed)

plt.subplot(222)
plt.imshow(imgGreen)

plt.subplot(223)
plt.imshow(imgBlue)

plt.subplot(224)
plt.imshow(umbIMG)
plt.show()

for x in range(0,h):
    for y in range(0,w):
        if image[x,y,0]>image[x,y,1] and image[x,y,0]>image[x,y,2]:
            imageOutput[x,y,0]=255
            imageOutput[x,y,1]=0
            imageOutput[x,y,2]=0
        elif image[x,y,1]>image[x,y,0] and image[x,y,1]>image[x,y,2]:
            imageOutput[x,y,0]=0
            imageOutput[x,y,1]=255
            imageOutput[x,y,2]=0
        else:
            imageOutput[x,y,0]=0
            imageOutput[x,y,1]=0
            imageOutput[x,y,2]=255
cv.imshow("Original",image)
cv.imshow("tres colores",imageOutput)

cv.waitKey(0)
cv.destroyAllWindows()
        