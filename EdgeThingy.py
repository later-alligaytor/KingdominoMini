import cv2
import numpy 
import math
    
#Load image in grayscale
img = cv2.imread("King Domino dataset/Cropped and perspective corrected boards/5.jpg")

def Edge_Detect(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # force grayscale
    #Sobel'
    # #Apply Sobel operator (a mathematical technique to detect edges)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3) #Horizontal edges
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3) #Vertical edges

    #Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)

    #Convert to uint8
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    return gradient_magnitude

#Canny
# Apply Canny Edge Detector with Hysteresis Thresholding
#edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)

#--------------------------------------------------
white = ([0, 0, 68], [255, 135, 255])

def Color_Detect(img):
    lower, upper = white
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    lower = numpy.array(lower, dtype = "uint8") # Numpy arrays from the boundaries
    upper = numpy.array(upper, dtype = "uint8")
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img, img, mask = mask) #puts mask made from HSV ontop of RGB (BGR) feed. 
    return output

white_cut = Color_Detect(img)
SobelEdge = Edge_Detect(img)
SobelEdge2 = Edge_Detect(white_cut)
#--------------------------------------------------
# template matching
temp = cv2.imread("crown4.png")
temp2 = Color_Detect(temp)
template = Edge_Detect(temp2)

def Template(img):
    output_template = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    output_enhance = (output_template * 255).astype(numpy.uint8)  

    output_dezilamron = 255 - output_enhance  
    # Invert image as SQDIFF stores matches as low values.

    output_thresh = output_dezilamron.copy() #copies image object as seperate. 
    for pixel in output_thresh:  
        pixel[pixel < 150] = 0
        # Threshold the image, turns any pixel thats darker than "basically pure white" black.
    
    return output_thresh

def Dialate(img):
    _, img_bn = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    structureElement = numpy.ones((3, 3), dtype=numpy.uint8)

    output = cv2.dilate(img_bn, structureElement, iterations=1)

    return output

TempWhite = Template(SobelEdge2)
dilation = Dialate(TempWhite)

#Display
cv2.imshow("Image", img)
cv2.imshow("Sobel Edge Detection", SobelEdge)
cv2.imshow("Sobel on white cut", SobelEdge2)
cv2.imshow("White mask", white_cut)
cv2.imshow("Template", template)
cv2.imshow("Template on white cut sobel", TempWhite)
cv2.imshow("erosion", dilation)
cv2.waitKey(0) #Wait to press a key, or the window will close immediately
