import cv2
import numpy as np

##work laptop
# imgPath = r"C:\Users\ismail.mohammed\TempScripts\manualOEE_OMR\OEE_forms\v1.6_TEMPLATE.jpg"

##home laptop
imgPath = r"C:\Users\Halima Mohmed\Documents\OEE_OMR\configFiles\blankForms\v2.1_blank0.jpg"

img = cv2.imread(imgPath)

# apply rotations, resizing etc
imgResize = cv2.resize(img, dsize=(int(img.shape[1]*0.4), int(img.shape[0]*0.4)), interpolation= cv2.INTER_AREA)
#imgResize = cv2.Resize(imgResize, cv2.Resize_90_CLOCKWISE)
imgGray = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
imgBlank = np.full_like(imgResize, 255)

ret, thresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# find all contours based off threshold image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

cornerConts = []
cornerCenters = []
# draw all contours on original image
for c in contours:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # find corner circles based of height, aspect ratio and x,y coords on page
    if 0.8 <= ar <= 1.2: #60 >= w >= 40 and 60 >= h >= 40 and  #and ((x > 500 and (y>400 or y<100)) or (x < 50 and (y>400 or y<100))):
        # compute enclosing circle to get center points
        (c_x, c_y), radius = cv2.minEnclosingCircle(c)
        c_x = np.round(c_x).astype('float32')
        c_y = np.round(c_y).astype('float32')
        cv2.drawContours(imgBlank, c, -1, (0, 0, 0))
        cornerConts.append(c)
        cornerCenters.append((c_x, c_y))
        #draw on blank to show correct detection
        cv2.circle(imgBlank, (c_x, c_y), 1, (0,0,255), 2)

#obtained corner centre points , next apply perspective transform, see: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
for cent in cornerCenters:
    print(cent)

#find circles using hough circles?
circles = cv2.HoughCircles(thresh,cv2.HOUGH_GRADIENT,1,50,
                            param1=50,param2=30,minRadius=5,maxRadius=50)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(imgResize,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(imgResize,(i[0],i[1]),2,(0,0,255),3)

#cv2.imshow("imgCircles", imgResize)
#cv2.imshow("imgThresh", thresh)
cv2.imshow("imgBlank", imgBlank)
cv2.waitKey(0)
