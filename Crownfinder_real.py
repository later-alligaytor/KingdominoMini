import cv2 as cv
import numpy as np

def find_crowns(image, template_paths, threshold=0.7):
    #template_paths is a list of paths to the crown templates
    #threshold is the match score from 0 to 1 (higher = stricter)
    #Returns a list of (x, y, w, h) boxes, one per crown found.
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    work = gray.copy() #Working copy where found crowns get painted white
    crowns = []

    for path in template_paths:
        template = cv.imread(path, cv.IMREAD_GRAYSCALE)
        h, w = template.shape

        result = cv.matchTemplate(work, template, cv.TM_CCOEFF_NORMED)

        while True:
            _, max_val, _, max_loc = cv.minMaxLoc(result)
            if max_val < threshold:
                break #No more matches good enough for this template

            x, y = max_loc
            crowns.append((x, y, w, h))

            #Paint the found crown white so no template finds it again
            work[y:y + h, x:x + w] = 255

            #Also wipe the score map around it so this loop doesn't repick it
            x0, y0 = max(0, x - w // 2), max(0, y - h // 2)
            result[y0:y + h // 2 + 1, x0:x + w // 2 + 1] = 0
        
        return crowns

if __name__ == "__main__":
    #Only runs when you execute this file directly, not when it is imported
    img = cv.imread("King Domino dataset/Cropped and perspective corrected boards/52.jpg")
    boxes = find_crowns(img, ["crown1.png", "crown.png"], threshold=0.6)

    print("Crowns found:", len(boxes))
    for x, y, w, h in boxes:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.imshow("Crowns", img)
        cv.waitKey(0)
        cv.destroyAllWindows()