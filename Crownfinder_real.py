import cv2 as cv
import numpy as np

ROTATIONS = [
    None,
    cv.ROTATE_90_CLOCKWISE,
    cv.ROTATE_180,
    cv.ROTATE_90_COUNTERCLOCKWISE,
]

def find_crowns(image, template_paths, threshold=0.7):
    #template_paths is a list of paths to the crown templates
    #threshold is the match score from 0 to 1 (higher = stricter)
    #Returns a list of (x, y, w, h) boxes, one per crown found.
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    work = gray.copy() #Working copy where found crowns get painted white
    crowns = []

    #Build one score map per template/rotation
    maps = [] #list of results (w, h)
    for path in template_paths:
        template = cv.imread(path, cv.IMREAD_GRAYSCALE)
        for rot in ROTATIONS:
            t = template if rot is None else cv.rotate(template, rot)
            h, w = t.shape #re-read each time: 90/270 degrees swap w and h
            result = cv.matchTemplate(work, t, cv.TM_CCOEFF_NORMED)
            maps.append((result, w, h))

        while True:
            #Find the best match across all maps
            best_val, best_i, best_loc = -1, -1, None
            for i, (result, _, _) in enumerate(maps):
                _, max_val, _, max_loc = cv.minMaxLoc(result)
                if max_val > best_val:
                    best_val, best_i, best_loc = max_val, i, max_loc

            if best_val < threshold:
                break #No more matches good enough for this template

            x, y = best_loc
            _, w, h = maps[best_i]
            crowns.append((x, y, w, h))

            #Paint the found crown white so no template finds it again
            work[y:y + h, x:x + w] = 255

            #Also wipe the score map around it so this loop doesn't repick it
            for result, tw, th in maps:
                x0, y0 = max(0, x - w // 2), max(0, y - h // 2)
                result[y0:y + h // 2 + 1, x0:x + w // 2 + 1] = 0
    return crowns

if __name__ == "__main__":
    #Only runs when you execute this file directly, not when it is imported
    img = cv.imread("King Domino dataset/Cropped and perspective corrected boards/52.jpg")
    boxes = find_crowns(img, ["crownupsidedown.png", "crown8.png"], threshold=0.7)
    
    print("Crowns found:", len(boxes))
    for x, y, w, h in boxes:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow("Crowns", img)
    cv.waitKey(0)
    cv.destroyAllWindows()