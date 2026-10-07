import cv2 as cv
from matplotlib import colors
from matplotlib.pyplot import hsv
import numpy as np
    
#Load image
image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/5.jpg")
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV) #HSV is better for color detection than RGB
cv.imshow("Image", image)
cv.waitKey(0) #Wait to press a key, or the window will close immediately
cv.imshow("HSV", hsv) #shows the HSV image, useful for changing the color ranges
cv.waitKey(0)

#Color ranges for red, blue, and green in HSV format (points needs to change, just using it as an example for now)
colors = {
    "red": {"low":(0, 120, 70), "high":(10, 255, 255), "point": "3"},
    "blue": {"low":(100, 150, 50), "high":(130, 255, 255), "point": "2"},
    "green": {"low":(20, 100, 100), "high":(35, 255, 255), "point": "1"},
}

MIN_AREA = 200 #Ignore small contours that are likely noise
kernel = np.ones((5, 5), np.uint8) #Kernel for morphological operations
total = 0 #Counter for total squares detected

for name, c in colors.items():
    #Build a mask: 255 where the color matches, 0 elsewhere
    mask = cv.inRange(hsv, np.array(c["low"]), np.array(c["high"]))

    #Remove noise and close small holes
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    #Find blobs (connected components)
    count, labels, stats, centroids = cv.connectedComponentsWithStats(mask)

    #Label 0 is the background, so start at 1
    pieces = sum(1 for i in range(1, count)
                 if stats[i, cv.CC_STAT_AREA] >= MIN_AREA)

    score = pieces * int(c["point"]) #Calculate score for this color
    total += score #Add to total score
    print(f"{name}: {pieces} pieces = {score} points")

print("Total:", total)