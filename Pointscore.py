import cv2 as cv
import numpy as np
from Crownfinder_real import find_crowns
import SliceTest

img = cv.imread("King Domino dataset/Cropped and perspective corrected boards/36.jpg")

count_crowns = find_crowns(img, ["crownupsidedown.png", "crown8.png"], threshold=0.7)

print(len(count_crowns))