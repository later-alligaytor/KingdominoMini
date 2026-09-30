from skimage import exposure
import sys
import cv2 as cv
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-r", required = True,
                help = "ratio", type=int, default = 800)
args = vars(ap.parse_args())

img = cv.imread("King Domino dataset\Cropped and perspective corrected boards\5.jpg")
