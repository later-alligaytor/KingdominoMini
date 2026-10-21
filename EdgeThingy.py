import cv2
import numpy 
import math
    
#img = cv2.imread("King Domino dataset/Cropped and perspective corrected boards/1.jpg")

def Edge_Detect(img): # Via. sobel
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # force grayscale
    # #Apply Sobel operator (a mathematical technique to detect edges)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3) #Horizontal edges
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3) #Vertical edges

    #Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)

    #Convert to uint8
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    
    
    #Canny
    # Apply Canny Edge Detector with Hysteresis Thresholding
    #edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)

    return gradient_magnitude

white = ([0, 0, 58], [255, 170, 255])

def Color_Detect(color, img): #detects white in image (used for the crowns)
    lower, upper = color
    imgg =cv2.convertScaleAbs(img, alpha=1.3, beta=0) # contrats (0.0 til 3.0) og brightness (-100 til 100)
    hsv = cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV_FULL)
    lower = numpy.array(lower, dtype = "uint8") # Numpy arrays from the boundaries
    upper = numpy.array(upper, dtype = "uint8")
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(imgg, imgg, mask = mask) #puts mask made from HSV ontop of image. 
    return output

# template matching
temp = cv2.imread("crown3.png")
tempWhite = Color_Detect(white, temp)
template = Edge_Detect(tempWhite)

def Template(img):
    h, w = img.shape[:2]
    output = numpy.zeros((h, w), dtype=numpy.uint8)
    ro_template = template.copy()

    for i in range(4):
        output_template = cv2.matchTemplate(img, ro_template, cv2.TM_SQDIFF_NORMED)
        output_enhance = (output_template * 255).astype(numpy.uint8)  

        output_dezilamron = 255 - output_enhance  
        # Invert image as SQDIFF stores matches as low values.

        output_thresh = output_dezilamron.copy() #copies image object as seperate. 
        for pixel in output_thresh:  
            pixel[pixel < 150] = 0
            # pixel[pixel > 150] = 255
            # Threshold the image, turns any pixel thats darker than "basically pure white" black.

        # Get dimensions
        th, tw = ro_template.shape[:2]
        # Pad the result to match the input dimensions
        outputM = numpy.zeros((h, w), dtype=numpy.uint8)
        # Offset matches by half the template dimensions
        y_offset = th // 2
        x_offset = tw // 2
        outputM[y_offset:y_offset + output_thresh.shape[0], x_offset:x_offset + output_thresh.shape[1]] = output_thresh

        lalala = cv2.bitwise_or(output, outputM, mask = None)
        output = lalala
        ro_template = cv2.rotate(ro_template, cv2.ROTATE_90_CLOCKWISE)

    # for pixel in output:  
    #     pixel[pixel < 220] = 0
    #     # pixel[pixel > 150] = 255
    #     # Threshold the image, turns any pixel thats darker than "basically pure white" black.
    return output


def Dialate(img): # Makes points bigger/easier to see.
    _, img_bn = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    structureElement = numpy.ones((4, 4), dtype=numpy.uint8)
    output = cv2.dilate(img_bn, structureElement, iterations=1)
    return output

#-------------------------------------------------------------
def GetCrowns(img):
    white_cut = Color_Detect(white, img) #get white
    SobelEdge = Edge_Detect(white_cut) #edge detect on the white
    TempWhite = Template(SobelEdge) #template on the edge detect
    dilation = Dialate(TempWhite) 
    cv2.imshow("thing",TempWhite)

    return dilation

def thing(int):
    img = cv2.imread(f"King Domino dataset/Cropped and perspective corrected boards/{int}.jpg")
    crowns = GetCrowns(img)
    crowns_col = cv2.cvtColor(crowns, cv2.COLOR_GRAY2BGR)
    together = numpy.concatenate((img, crowns_col), axis=1)

    # cv2.imshow(f"Image {int}", img)
    # cv2.imshow(f"Detection {int}", crowns)
    cv2.imshow(f"Test {int}", together)
    
# thing(1)
# thing(2)
thing(3)
thing(4)
thing(7)
thing(10)
thing(18)
cv2.imshow("template",template)

#Display
# cv2.imshow("Image", img)
# cv2.imshow("White mask", white_cut)
# cv2.imshow("Template", template)
# cv2.imshow("Template on white cut sobel", TempWhite)
# cv2.imshow("erosion", dilation)
cv2.waitKey(0) #Wait to press a key, or the window will close immediately
