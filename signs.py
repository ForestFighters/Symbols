#!/usr/bin/env python

import numpy as np
import cv2 as cv

# Class of 
class symbol():
    def __init__(self, name, img):
        self.img = img
        self.name = name
   
  
def orderCorners( corners ):
    # Empty rectangle for our results
    rect = np.zeros((4,2), dtype = "float32")
    
    # Our convert our supplied list for processing 
    corners = np.array(corners, dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = corners.sum(axis = 1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(corners, axis = 1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    # return the ordered coordinates
    return rect
      
# Read in our reference images
def readRefImages():

    symbols = []    
    img = cv.imread("arrowL.jpg", cv.IMREAD_GRAYSCALE)
    if img is None:
        return -1, None
    img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)[1]
    symbols.append(symbol("Left 90", img))    
        
    img = cv.imread("arrowR.jpg", cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)[1]
    symbols.append(symbol("Right 90", img))

    img = cv.imread("arrowT.jpg", cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)[1]
    symbols.append(symbol("Turn Around", img))

    img = cv.imread("arrowB.jpg", cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)[1]
    symbols.append(symbol("Ball", img))

    img = cv.imread("arrowL45.jpg", cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)[1]
    symbols.append(symbol("Left 45", img))

    img = cv.imread("arrowR45.jpg", cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)[1]
    symbols.append(symbol("Right 45",img))

    img = cv.imread("arrowStop.jpg", cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)[1]
    symbols.append(symbol("Stop", img))

    img = cv.imread("arrowGo.jpg", cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)[1]    
    symbols.append(symbol("Go", img))

    return symbols


def main():
    # Fire up the camera
    cap = cv.VideoCapture(0)
    
    # Create some windows so we can see the results
    cv.namedWindow("Original")
    cv.namedWindow("Sign")
    # Uncomment when viewing process steps for debugging
    # cv.namedWindow("Debug")
    
    # Cache some symbols we want to recognise
    symbols = readRefImages()
    if symbols is None:
        print("Error reading reference images")
                    
    while True:
        # Grab our image from the camera
        _status, img = cap.read()        
        
        # Convert to grey scale and remove noise
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)        
        grey = cv.GaussianBlur( grey, (9,9), cv.BORDER_DEFAULT )
        # cv.imshow("Debug",grey)

        # Find the edges 
        threshold = 12
        canny_output = cv.Canny(grey, threshold, threshold * 3)
        # cv.imshow("Debug",canny_output)
        
        # Find the contours in the image from the canny output
        contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                
        # Find those that look like rectangles (have 4 edges)
        for contour in contours:
            # Look for a rectangle
            approxRect = cv.approxPolyDP(contour, cv.arcLength( contour, True) * 0.05, True)            
            if len(approxRect) == 4:                
                area = cv.contourArea(contour)
                # Process rectangles with a resonable area
                if area > 10000:                                        
                    # Grab the corner verticies (Place a circle at each for debugging)
                    corners = []                    
                    vertex = approxRect[0][0]                
                    cv.circle(img, (vertex[0],vertex[1]), 2, (0, 0, 255), -1)
                    corners.append((vertex[0],vertex[1]))
                    
                    vertex = approxRect[1][0]
                    cv.circle(img, (vertex[0],vertex[1]), 2, (0, 0, 255), -1)
                    corners.append((vertex[0],vertex[1]))
                    
                    vertex = approxRect[2][0]
                    cv.circle(img, (vertex[0],vertex[1]), 2, (0, 0, 255), -1)
                    corners.append((vertex[0],vertex[1]))
                    
                    vertex = approxRect[3][0]
                    cv.circle(img, (vertex[0],vertex[1]), 2, (0, 0, 255), -1)
                    corners.append((vertex[0],vertex[1]))
                    
                    # Sort the corners of our found rectangle into a known order tl, tr, br, bl
                    corners = orderCorners(corners)                    
                    
                    # Define the destination image                    
                    correctedImg = np.zeros((195, 271, 3), np.uint8)
                    # and its corners                     
                    quad = np.array([[0,0],[271, 0],[271,195],[0,195]], dtype = "float32")                    
                    
                    # Compute the transform to get from found rectange to to our corrected rectange
                    transmtx = cv.getPerspectiveTransform(corners, quad)
                    # and apply it to the grey scale version
                    warped = cv.warpPerspective(grey, transmtx, (271,195))
                    
                    # Threshold to mono chome
                    retval, warped = cv.threshold(warped, 150,255,cv.THRESH_BINARY)
                    # cv.imshow("Debug",warped)
                     
                    # Now fine the Symbol that most closely matches the transformed image
                    minDiff = 12000
                    match = -1

                    for i in range(8):                        
                        diffImg = cv.bitwise_xor(warped, symbols[i].img)
                        diff = cv.countNonZero(diffImg)
                        if diff < minDiff:
                            minDiff = diff
                            match = i
                        
                    # If one was located then display it's name on the original and display the symbol
                    if match != -1:
                        print("Area = ",area)
                        cv.putText(img, symbols[match].name, (320, 30), 1, 2, (0, 255, 0), 2)
                        cv.imshow("Sign", symbols[match].img)
                    
                    
        # Give the OCV a chance to process the display
        cv.imshow("Original",img)
        k = cv.waitKey(10)

        if k == 27:
            break


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
    