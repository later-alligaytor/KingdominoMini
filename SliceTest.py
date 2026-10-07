import cv2
import numpy
import math

img = cv2.imread(r"King Domino dataset\Cropped and perspective corrected boards\14.jpg") # the "r" is for raw string. Otherwise it might interpret things in the path as commands
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

#--------------------------------------------------------teste tile type ting

plainsOut = numpy.zeros((5, 5)) #en matrix vi gør blob/grassfire ting på senere. (viser hvilke tiles er grasslands/plains)
forestOut = numpy.zeros((5, 5))
cornOut = numpy.zeros((5, 5))

# Color boundaries in HSV (eller HSB) (lower, upper)
plains = ([49, 0, 136], [93, 255, 200])
forest = ([49, 0, 0], [93, 255, 79])
corn = ([32, 0, 154], [41, 255, 251])

# Color detection ud fra boundaries. Laver sort/hvid binary billed af billedet(tile) man putter ind. 
def color_Detect(color,tile):
    lower, upper = color
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV_FULL)
    # create NumPy arrays from the boundaries
    lower = numpy.array(lower, dtype = "uint8")
    upper = numpy.array(upper, dtype = "uint8")
    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(hsv, lower, upper)
    return mask

# def der checker om en spicefik tile er en spicefik type, og markerer 
def thing(type,color,row,number):
    tile = Slice(row,number)
    tile_detect = color_Detect(color,tile)
    tileHeight = tile.shape[0]
    tileWidth = tile.shape[1]
    
    value = math.floor(numpy.sum(tileHeight*tileWidth/3)) 
    pixel_hits = numpy.sum(tile_detect == 255) # extracting only white pixels 

    #if pixel_hits > value: #hvis antallet af pixel hits er stører en 1/3 af alle pixels i billedet...
    #    type[row-1,number-1] = 1
    #    check = 1
    check = 1 if pixel_hits > value else 0
    return check

def tingting(type,color):
    for y, row in enumerate(type): # loops over output data points to populate it. 
            for x, pixel in enumerate(row):
                check = thing(type,color,y+1,x+1)
                type[y, x] = check 
    return type

#outblib = thing(cornOut,corn, 1,1)
#outblib = thing(cornOut,corn, 1,2)
#outblib = thing(cornOut,corn, 4,4)
#outblib = thing(cornOut,corn, 4,5)
#outblib = thing(cornOut,corn, 3,5)
#outblib = thing(cornOut,corn, 2,5)
#outblib = thing(plainsOut,plains, 2,2)
#outblib = thing(plainsOut,plains, 4,4)
#outblib = thing(plainsOut,plains, 4,5)

#outblib6 = thing(forestOut,forest, 5,3)
#outblib6 = thing(forestOut,forest, 5,4)
#outblib6 = thing(forestOut,forest, 4,3)
#outblib6 = thing(forestOut,forest, 1,2)

#outblibUP = cv2.resize(outblib, dsize=(outblib.shape[0]*10, outblib.shape[1]*10)) #sizes up
#outblib6UP = cv2.resize(outblib6, dsize=(outblib6.shape[0]*10, outblib6.shape[1]*10)) #sizes up

aba = tingting(forestOut,forest)
bab = tingting(cornOut,corn)
abaUP = cv2.resize(aba, dsize=(aba.shape[0]*40, aba.shape[1]*40)) #sizes up
babUP = cv2.resize(bab, dsize=(bab.shape[0]*40, bab.shape[1]*40)) #sizes up

cv2.imshow("Img", img)
#cv2.imshow("Slice", slice1)
#cv2.imshow("detection", test)
#cv2.imshow("corn", outblibUP)
#cv2.imshow("forest", outblib6UP)
cv2.imshow("forest", abaUP)
cv2.imshow("corn", babUP)
cv2.waitKey(0)