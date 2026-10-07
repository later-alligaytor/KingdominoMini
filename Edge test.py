import cv2 as cv
import numpy as np
import tempfile

def extract_squares(image_path, rows, columns):
    #Trying to use the same as a chessboard, may not work
    inner_rows = rows - 1
    inner_columns = columns - 1
    #Load image
    image = cv.imread("King Domino dataset\Cropped and perspective corrected boards\5.jpg")

    #Convert the image to greyscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #Find the inner corners
    found, corners = cv.find4QuadCornerSubpix(gray, (inner_row, inner_columns), None)

    if found:
        #Refine the corner locations
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1))

        #Create a 2D list to store file names of squares (needed?)
        