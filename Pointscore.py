import cv2 as cv
import numpy as np
from Crownfinder_real import find_crowns
import SliceTest

def GetScore(int):
    img = cv.imread(f"King Domino dataset/Cropped and perspective corrected boards/{int}.jpg")
    # cv.imshow(f"Board {int}", img) 
    score = 0
    print(f"For board {int};")

    for i, color in enumerate(SliceTest.colors):
        a = SliceTest.Board_check(img,SliceTest.types[i],color) #color kan også erstats med colors[i]
        b = SliceTest.connected_blobs(a)

        blobs = np.unique(b)
        blobs = blobs[blobs != 0] 
        # gets the unique numbers in matrix, and removes 0's. Fx [1, 1, 0, 2, 3, 4, 4, 0,] -> [0, 1, 2, 3, 4] -> [1, 2, 3, 4]
    
        for i, blob in enumerate(blobs):
            crowns_count = []
            blob_area = (b == blob)
            for y, row in enumerate(blob_area):
                for x, number in enumerate(row):
                    if blob_area[y, x] == 0:
                        continue #skips blanks. 
                    tile = SliceTest.Slice(img, y+1, x+1)
                    crowns = find_crowns(tile, ["crownupsidedown.png", "crown8.png"], threshold=0.7)
                    crowns_count.extend(crowns)

            num_tiles = np.count_nonzero(b == blob)
            num_crowns = len(crowns_count)
            print(f"{color[2]} ID {i+1} has {num_tiles} tiles and {num_crowns} crowns, and has an area score of {num_tiles*num_crowns}") #i+1 kan også erstates af "blob"
            score += num_tiles*num_crowns

    print(f"Final score is {score}") #i+1 kan også erstates af "blob"
    print("")

j = 0
for i in range(75):
    ana = [1,2,3,9,16,18,20,25,37,42]
    if i in ana:
        GetScore(i)
        j+=1

print(f"Analysed {j} boards")
# cv.waitKey(0)