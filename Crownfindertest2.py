import cv2 as cv
from matplotlib import colors
from matplotlib.pyplot import hsv
import numpy as np
    
#Load image in grayscale
img = cv.imread("King Domino dataset/Cropped and perspective corrected boards/14.jpg", cv.IMREAD_GRAYSCALE)

#Apply Sobel operator (a mathematical technique to detect edges)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3) #Horizontal edges
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3) #Vertical edges

#Compute gradient magnitude
gradient_magnitude = cv.magnitude(sobelx, sobely)

#Convert to uint8
gradient_magnitude = cv.convertScaleAbs(gradient_magnitude)

#Display result
cv.imshow("Sobel Edge Detection", gradient_magnitude)
cv.waitKey(0) #Wait to press a key, or the window will close immediately

## template matching
#template = cv.imread("crown2.png")
#
#output_template = cv.matchTemplate(hsv, template, cv.TM_SQDIFF)
#output_normalized = ((output_template / np.max(output_template)) * 255).astype(np.uint8)  
## normalized version of template matching operation SQDIFF. Like above but with normalization.
#
#output_dezilamron = 255 - output_normalized  
## Invert image as matchTemplate stores matches as low values.
#
#output_thresh = output_dezilamron.copy() #copies image object as seperate. 
#for pixel in output_thresh:  
#    pixel[pixel < 240] = 0
#    # Threshold the image, turns any pixel thats darker than "basically pure white" black.
#
##Color ranges for red, blue, and green in HSV format (points needs to change, just using it as an example for now)
#colors = {
#    "red": {"low":(0, 120, 70), "high":(10, 255, 255), "point": "3"},
#    "blue": {"low":(100, 150, 50), "high":(130, 255, 255), "point": "2"},
#    "green": {"low":(20, 100, 100), "high":(35, 255, 255), "point": "1"},
#}
#
#MIN_AREA = 200 #Ignore small contours that are likely noise
#kernel = np.ones((5, 5), np.uint8) #Kernel for morphological operations
#total = 0 #Counter for total squares detected
#
#for name, c in colors.items():
#    #Build a mask: 255 where the color matches, 0 elsewhere
#    mask = cv.inRange(hsv, np.array(c["low"]), np.array(c["high"]))
#
#
#    #Remove noise and close small holes
#    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
#    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
#
#    #Find connected components
#    count, labels, stats, centroids = cv.connectedComponentsWithStats(mask)
#
#    #Label 0 is the background, so start at 1
#    pieces = sum(1 for i in range(1, count)
#                 if stats[i, cv.CC_STAT_AREA] >= MIN_AREA)
#
#print("Total:", total)
#
#cv.imshow("template only", output_template)
#cv.imshow("inverse", output_dezilamron)
#cv.imshow("Threshold", output_thresh)
#cv.waitKey(0) # if not here, will instantly destroy windows after loading. Waits until a key is pressed to destroy.