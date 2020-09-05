import cv2
import numpy as np

imgPath = r"C:\Users\ismail.mohammed\TempScripts\manualOEE_OMR\OEE_forms\v1.6_TEMPLATE.jpg"
img = cv2.imread(imgPath)
imgSmall = cv2.resize(img,None, fx= 0.3, fy=0.3)

imgGray = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2GRAY)
ret, imgThresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_OTSU)
imgCanny = cv2.Canny(imgGray, 200,400)
imgBlank = np.full_like(imgSmall, 255)
print(ret)
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:

    peri = cv2.arcLength(c, True)
    shape = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(shape) != 1:
        cv2.drawContours(imgBlank, c, -1, (0, 0, 255))

cv2.imshow("imgCanny", imgCanny)
cv2.imshow("imgBlank", imgBlank)
cv2.imshow("imgThresh", imgThresh)

cv2.waitKey(10000000)
