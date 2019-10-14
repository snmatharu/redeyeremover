import cv2
import numpy as np


def fillHoles(mask):
    maskFloodfill = mask.copy()
    h, w = maskFloodfill.shape[:2]
    maskTemp = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
    mask2 = cv2.bitwise_not(maskFloodfill)
    return mask2 | mask

if __name__ == '__main__' :

    img = cv2.imread("red_eyes3.jpg", cv2.IMREAD_COLOR)
    imgOut = img.copy()
    eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    eyes = eyesCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(100, 100))
    for (x, y, w, h) in eyes:

        eye = img[y:y+h, x:x+w]
        b = eye[:, :, 0]
        g = eye[:, :, 1]
        r = eye[:, :, 2]
        bg = cv2.add(b, g)
        mask = (r > 150) &  (r > bg)
        mask = mask.astype(np.uint8)*255
        mask = fillHoles(mask)
        mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)
        mean = bg / 2
        mask = mask.astype(np.bool)[:, :, np.newaxis]
        mean = mean[:, :, np.newaxis]
        eyeOut = eye.copy()
        eyeOut = np.where(mask, mean, eyeOut)
        imgOut[y:y+h, x:x+w, :] = eyeOut
    cv2.imshow('Red Eyes', img)
    cv2.imshow('Red Eyes Removed', imgOut)
    cv2.waitKey(0)
