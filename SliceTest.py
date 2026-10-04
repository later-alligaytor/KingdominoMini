import cv2
import numpy
import math

img = cv2.imread(r"King Domino dataset\Cropped and perspective corrected boards\1.jpg") # the "r" is for raw string. Otherwise it might interpret things in the path as commands
imgHeight = img.shape[0] # or images max y
imgWidth = img.shape[1] # or max x

tileY = math.floor(imgHeight/5)
tileX = math.floor(imgWidth/5)

def Slice(row,number):
    slice = img[tileY*row-tileY: tileY*row , tileX*number-tileX: tileX*number] #slices from [min y : max y , min x : max x]. Empty means it uses the default.
    return slice

slice1 = Slice(1,1)
slice2 = Slice(1,2)

Hori = numpy.concatenate((slice1, slice2), axis=1)

cv2.imshow("Img", img)
cv2.imshow("Slice", slice1)
cv2.imshow("Slice2", slice2)
cv2.imshow("Hori", Hori)
cv2.waitKey(0)