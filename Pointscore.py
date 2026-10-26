import cv2 as cv
import numpy as np
from Crownfinder_real import find_crowns
import SliceTest

img = cv.imread("King Domino dataset/Cropped and perspective corrected boards/36.jpg")
for y, row in enumerate(img):
        for x, pixel in enumerate(row):
            tile = SliceTest.Slice(img,y+1,x+1).astype(np.uint8)

count_crowns = find_crowns(tile, ["crownupsidedown.png", "crown8.png"], threshold=0.7)

print(len(count_crowns))