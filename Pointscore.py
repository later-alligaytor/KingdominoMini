import cv2 as cv
import numpy as np
from Crownfinder_real import find_crowns
from SliceTest import types, colors, Board_check

total_crowns = 0
img = cv.imread("King Domino dataset/Cropped and perspective corrected boards/36.jpg")
count_crowns = find_crowns(img, ["crownupsidedown.png", "crown8.png"], threshold=0.7)

for row in range(1, 6):
    for col in range(1, 6):
        tile = SliceTest.Slice(img, row, col)
        count_crowns = find_crowns(tile, ["crownupsidedown.png", "crown8.png"], threshold=0.7)
        print(f"Tile ({row}, {col}): {len(count_crowns)} crowns")
        # total_crowns += len(count_crowns)

# count_crowns = find_crowns(tile, ["crownupsidedown.png", "crown8.png"], threshold=0.7)
# print(len(count_crowns))
# print(count_crowns)