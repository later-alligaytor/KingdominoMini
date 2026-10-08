import cv2
import numpy
import math

img = cv2.imread(r"King Domino dataset\Cropped and perspective corrected boards\2.jpg") 
# the "r" is for raw string. Otherwise it might interpret things in the path as commands

#region: Old "get size og image and tile size. Moved into Slice()"
#imgHeight = img.shape[0] # or images max y. 
#imgWidth = img.shape[1] # or max x

#tileY = math.floor(imgHeight/5)
#tileX = math.floor(imgWidth/5)
#endregion

#makes a slice of 1 tile from board image.
def Slice(board,row,number):
    tileY = math.floor(board.shape[0]/5)
    tileX = math.floor(board.shape[1]/5)
    slice = board[tileY*row-tileY: tileY*row , tileX*number-tileX: tileX*number] #slices from [min y : max y , min x : max x]. Empty means it uses the default.
    return slice

#Hori = numpy.concatenate((slice1, slice2), axis=1)

# matricer vi gør blob/grassfire ting på senere. (viser hvilke tiles er af hver type)
plainsOut = numpy.zeros((5, 5))
forestOut = numpy.zeros((5, 5))
cornOut = numpy.zeros((5, 5))
caveOut = numpy.zeros((5, 5))


# Color boundaries i HSV (eller HSB) (lower, upper)
plains = ([49, 0, 136], [93, 255, 200])
forest = ([49, 0, 0], [93, 255, 79])
corn = ([32, 0, 154], [41, 255, 251])
cave = ([0, 0, 0], [255, 255, 25])

# Color detection ud fra boundaries. Laver sort/hvid binary billed af et tile-slice. 
def Color_Detect(color,tile):
    lower, upper = color
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV_FULL)
    lower = numpy.array(lower, dtype = "uint8") # Numpy arrays from the boundaries
    upper = numpy.array(upper, dtype = "uint8")
    mask = cv2.inRange(hsv, lower, upper)
    return mask

# Checker om en spicefik tile er en spicefik type, og retunerer enten 1 eller 0.  
def Tile_check(board,color,row,number):
    tile = Slice(board,row,number)
    tile_detect = Color_Detect(color,tile)
    tileHeight = tile.shape[0]
    tileWidth = tile.shape[1]
    
    value = math.floor(numpy.sum(tileHeight*tileWidth/4)) 
    if (color == cave):
        value = math.floor(numpy.sum(tileHeight*tileWidth/5)) # dem med mange kroner er sværer at detecte.
    pixel_hits = numpy.sum(tile_detect == 255) # extracts only white pixels 

    check = 1 if pixel_hits > value else 0
    return check

# Redigerer type-matricen ud fra resultaternde af tile_check board-whide.
def Board_check(board,type,color):
    for y, row in enumerate(type):
            for x, pixel in enumerate(row):
                check = Tile_check(board,color,y+1,x+1)
                type[y, x] = check 
    return type

# Gør type-matrix billede stører (ellers er de kun 5x5 pixels store)
def Enlarge(img):
     imgUP = cv2.resize(img, dsize=(img.shape[0]*40, img.shape[1]*40)) #sizes up
     return imgUP

#outblib = thing(cornOut,corn, 1,1)

aba = Board_check(img,forestOut,forest)
bab = Board_check(img,cornOut,corn)
cab = Board_check(img,caveOut,cave)
abaUP = Enlarge(aba)
babUP = Enlarge(bab)
cabUP = Enlarge(cab)

cv2.imshow("Img", img)
#cv2.imshow("Slice", slice1)
#cv2.imshow("detection", test)
#cv2.imshow("corn", outblibUP)
#cv2.imshow("forest", outblib6UP)
cv2.imshow("forest", abaUP)
cv2.imshow("corn", babUP)
cv2.imshow("cave", cabUP)
cv2.waitKey(0)