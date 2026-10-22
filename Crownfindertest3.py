import cv2 as cv
import numpy as np
    
#Load image
image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/52.jpg")
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV) #HSV is better for color detection than RGB
output = image.copy() #Draw on the copy so the original stays clean

#Color ranges for red, blue, and green in HSV format - change numbers to get the crowns more clear
CROWN_LOW = (15, 111, 111)
CROWN_HIGH = (85, 130, 255)

MIN_AREA = 20 #since the crowns are small we should start low and slowly increase
kernel = np.ones((3, 3), np.uint8) #A small kernel so the crowns survive

mask = cv.inRange(hsv, np.array(CROWN_LOW), np.array(CROWN_HIGH))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel) #Should only close holes, no opens

cv.imshow("Crown mask", mask) #Check this first, are the crowns white?

count, labels, stats, centroids = cv.connectedComponentsWithStats(mask)

crowns = []
for i in range(1, count):
    area = stats[i, cv.CC_STAT_AREA]
    print(f"blob {i}: area={area}") #This helps pick a minimum area
    if area < MIN_AREA:
        continue
    cx, cy = map(int, centroids[i])
    crowns.append((cx, cy))
    x, y, w, h = stats[1, :4]
    cv.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

print("Crowns found:", len(crowns))      
cv.imshow("Results", output)
cv.waitKey(0)
cv.destroyAllWindows()