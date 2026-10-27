import cv2
import numpy
import math

#makes a slice of 1 tile from board image.
def Slice(board,row,number):
    tileY = math.floor(board.shape[0]/5)
    tileX = math.floor(board.shape[1]/5)
    slice = board[tileY*row-tileY: tileY*row , tileX*number-tileX: tileX*number] #slices from [min y : max y , min x : max x]. Empty means it uses the default.
    return slice

# matricer vi gør blob/grassfire ting på senere. (viser hvilke tiles er af hver type)
# plainsOut = numpy.zeros((5, 5))
# forestOut = numpy.zeros((5, 5))
# cornOut = numpy.zeros((5, 5))
# caveOut = numpy.zeros((5, 5))
types = [
	numpy.zeros((5, 5)), # 0 Grass plains
	numpy.zeros((5, 5)),    # 1 Forests
	numpy.zeros((5, 5)), # 2 Corn fields
	numpy.zeros((5, 5)),# 3 Oceans
	numpy.zeros((5, 5)),  # 4 Desserts
	numpy.zeros((5, 5)),    # 5 Caves
]

colors = [
	([49, 0, 100], [87, 255, 255], "Grass"), # 0 Grass plains
	([49, 0, 0], [93, 255, 79], "Forest"),    # 1 Forests
	([32, 0, 154], [41, 255, 251], "Corn"), # 2 Corn fields
	([129, 85, 65], [182, 255, 223], "Ocean"),# 3 Oceans
	([20, 0, 50], [45, 220, 140], "Dessert"),  # 4 Desserts
	([0, 0, 0], [35, 140, 50], "Cave"),    # 5 Caves
]
# Color boundaries i HSV (eller HSB) (lower, upper)
# plains = ([49, 0, 136], [93, 255, 200])
# forest = ([49, 0, 0], [93, 255, 79])
# corn = ([32, 0, 154], [41, 255, 251])
# cave = ([0, 0, 0], [255, 255, 25])

# Color detection ud fra boundaries. Laver sort/hvid binary billed af et tile-slice. 
def Color_Detect(color,tile):
    lower, upper, name = color
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV_FULL)
    lower = numpy.array(lower, dtype = "uint8") # Numpy arrays from the boundaries
    upper = numpy.array(upper, dtype = "uint8")
    mask = cv2.inRange(hsv, lower, upper)
    if (color == colors[5]): # if caves.
        mask2 = cv2.inRange(hsv, numpy.array([145, 0, 0], dtype = "uint8"), numpy.array([255, 140, 50], dtype = "uint8"))
        mask = cv2.bitwise_or(mask, mask2)
    return mask

# Checker om en spicefik tile er en spicefik type, og retunerer enten 1 eller 0.  
def Tile_check(board,color,row,number):
    tile = Slice(board,row,number)
    tile_detect = Color_Detect(color,tile)
    tileHeight = tile.shape[0]
    tileWidth = tile.shape[1]
    
    value = math.floor(numpy.sum(tileHeight*tileWidth/4)) 
    if (color == colors[4]): # if Dessert.
        value = math.floor(numpy.sum(tileHeight*tileWidth/2)) # er nem at detecte som starter tile (det må den ikke)
    
    if (color == colors[5]): # if caves.
        value = math.floor(numpy.sum(tileHeight*tileWidth/6)) # dem med mange kroner er sværer at detecte.

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

# Gør matrix billede stører (ellers er de kun 5x5 pixels store)
def Enlarge(blobsimg, board):
     imgUP = cv2.resize(blobsimg, dsize=(board.shape[0], board.shape[1],), interpolation = cv2.INTER_NEAREST) #sizes up
     return imgUP

#blob testing------------------------------------------------------------------------------------
#Make blobs
def connected_blobs(array): #blobs af connected felter.
    result = numpy.zeros(array.shape)
    id = 1
    for y, row in enumerate(array):
            for x, pixel in enumerate(row):
                if array[y, x] == 0 or result[y, x] != 0:
                    continue #skips blanks and already burned pixels. 
                burn_queue = [(y, x)]
                while len(burn_queue) > 0:
                    y, x = burn_queue.pop()
                    if result[y,x]!= 0:
                        continue #skips burned pixels.
                    
                    result[y, x] = id

                    if x+1 < array.shape[1] and array[y, x+1] != 0 and result[y, x+1] == 0:
                        burn_queue.append((y, x + 1))
                    if y+1 < array.shape[0] and array[y+1, x] != 0 and result[y+1, x] == 0:
                        burn_queue.append((y + 1, x))
                    if x-1 >= 0 and array[y, x-1] != 0 and result[y, x-1] == 0:
                        burn_queue.append((y, x - 1))
                    if y-1 >= 0 and array[y-1, x] != 0 and result[y-1, x] == 0:
                        burn_queue.append((y - 1, x))
                id += 1
    return result

#Get num of tiles in each blob
def blob_tiles(blobsimg, color):
    blobs = numpy.unique(blobsimg)
    blobs = blobs[blobs != 0] # gets the unique numbers in matrix, and removes 0's. Fx [1, 1, 0, 2, 3, 4, 4, 0,] -> [0, 1, 2, 3, 4] -> [1, 2, 3, 4]
    print(blobs)
    
    for i, blob in enumerate(blobs):
        print(f"number of {color[2]} tiles in id {i+1}: {numpy.count_nonzero(blobsimg == blob)}") #i+1 kan også erstates af "blob"

#Get a blob
def GetBlob(blobsimg, id, board):
    if not numpy.any(blobsimg == id):
        print(f"Array does not have a blob with ID {id}")
    else:
        blobling = Enlarge((blobsimg == id).astype(numpy.uint8), board)* 255
        blobling = (blobsimg == id).astype(numpy.uint8)
        # cv2.imshow(f"Blob {id}",blobling)
    return blobling

#------------------------------------------------------------------------------------------

def BoardAnalysis(int):
    img = cv2.imread(f"King Domino dataset/Cropped and perspective corrected boards/{int}.jpg")
    for i, color in enumerate(colors):
        a = Board_check(img,types[i],color) #color kan også erstats med colors[i]
        b = connected_blobs(a)

        aUP = Enlarge(a, img)
        cv2.imshow("Board", img) 
        cv2.imshow(f"{color[2]}", aUP) #color[2] kan også erstates med colors[i][2]

# BoardAnalysis(5)
# cv2.waitKey(0)