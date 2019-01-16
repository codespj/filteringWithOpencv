import sys
import cv2
import numpy as np
import math as math

if len(sys.argv) == 2:
    imageFileName = sys.argv[1]   
else:
    camera = cv2.VideoCapture(0)
    while True:            
        return_value,image = camera.read()
        cv2.imshow('Capture',image)
        if cv2.waitKey(1) != -1:
            cv2.imwrite('temp.jpg',image)                
            break
    camera.release()
    cv2.destroyAllWindows()
    imageFileName = 'temp.jpg'
img = cv2.imread(imageFileName)
img = cv2.resize(img,(1024,600))
cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
cv2.imshow('Capture',img)
red = True
blue = False
green = False
while True:
    c = cv2.waitKey(0) & 0xFF
    if c == ord('i'):
        img = cv2.imread(imageFileName)
        img = cv2.resize(img,(1024,600))
        cv2.imshow('Capture',img)
    elif c == ord('w'):
        cv2.imwrite('out.jpg',img) 
    elif c == ord('g'):
        img = cv2.imread(imageFileName)
        img = cv2.resize(img,(1024,600)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Capture',img)
    elif c == ord('G'):
        img = cv2.imread(imageFileName)   
        img = cv2.resize(img,(1024,600))     
        grayscle = img[:,:,0]
        cv2.imshow('Capture',grayscle)
    elif c == ord('c'):
        img = cv2.imread(imageFileName)
        img = cv2.resize(img,(1024,600))
        b,g,r = cv2.split(img)
        if red:
             img = r.copy()
             red = False
             blue = True
        elif blue:
             img = b.copy()
             blue = False
             green = True
        elif green:
             img = g.copy()
             green = False
             red = True
        cv2.imshow('Capture', img) 
    elif c == ord('s'):
        smooth = cv2.blur(img,(10,10))
        cv2.imshow('Capture',smooth)
    elif c == ord('S'):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Making Smoothing Filter
        kernel = np.ones((21,21),np.float32)/(21*21)
        #Convolving the Image - (Here It will build the kernel and filter image using kernel)
        finalImage = cv2.filter2D(img_gray, -1, kernel)
        cv2.imshow("Capture",finalImage)
    elif c == ord('d'):
        factor = 0.2
        newsize = (int(img.shape[0] * factor), int(img.shape[1] * factor))
        img = cv2.resize(img, newsize)
        cv2.imshow('Capture',img)
    elif c == ord('D'):
        factor = 0.2
        newsize = (int(img.shape[0] * factor), int(img.shape[1] * factor))
        img = cv2.resize(img, newsize)
        smooth = cv2.blur(img,(10,10))
        cv2.imshow('Capture',smooth)
    elif c == ord('x'):
        img = cv2.imread(imageFileName)
        img = cv2.resize(img,(1024,600)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Sobel(img,cv2.CV_64F,1,0)
        img = np.absolute(img)
        img = np.uint8(255*img/np.max(img))
        cv2.imshow('Capture',img)
    elif c == ord('y'):
        img = cv2.imread(imageFileName) 
        img = cv2.resize(img,(1024,600))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Sobel(img,cv2.CV_64F,0,1)
        img = np.absolute(img)
        img = np.uint8(255*img/np.max(img))
        cv2.imshow('Capture',img) 
    elif c == ord('m'):
        img = cv2.imread(imageFileName)  
        img = cv2.resize(img,(1024,600))
        img = cv2.Sobel(img,cv2.CV_64F,1,1)
        img = np.absolute(img)
        img = np.uint8(255*img/np.max(img))
        img = cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
        cv2.imshow('Capture',img) 
    elif c == ord('r'):
        img = cv2.imread(imageFileName)  
        img = cv2.resize(img,(1024,600))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img_gray.shape[:2]
        image_center = (width / 2, height / 2)
        img = cv2.getRotationMatrix2D(image_center, 45, 1)
        radians = math.radians(45)
        sin = math.sin(radians)
        cos = math.cos(radians)
        bound_w = int((height * abs(sin)) + (width * abs(cos)))
        bound_h = int((height * abs(cos)) + (width * abs(sin)))
        img[0, 2] += ((bound_w / 2) - image_center[0])
        img[1, 2] += ((bound_h / 2) - image_center[1])
        img = cv2.warpAffine(img_gray, img, (bound_w, bound_h))
        cv2.imshow("Capture",img)
    elif c == ord('h'):
        img = cv2.imread(imageFileName)  
        img = cv2.resize(img,(1024,600))
        font                   = cv2.FONT_HERSHEY_PLAIN
        fontScale              = 0.5
        fontColor              = (0x3f, 0x2c, 0x36)
        lineType               = 1

        text = "\n Program :- The Application is used to capture image from the continuous camera mode or \n"
        text += "from file system \n and perform some operations on them depending on the user input."
        text += "\n Command Line Argument - python test.py (for normal camera mode) \n"
        text += "&& python test.py background.png (For file)\n"
        text += "\n Supported Keys :- \n"
        text += "i - Reload original image \n"
        text += "w - Write to file \n"
        text += "g - Convert to Grayscale \n"
        text += "G - Convert to Grayscale (Custom Implementation) \n"
        text += "c - Cycle through color channels \n"
        text += "s - Convert to Grayscle and Smooth\n"
        text += "S - Convert to Grayscle and Smooth (Custom) \n"
        text += "d - Downsample by factor 2 \n"
        text += "D - Downsample by factor 2 with smoothing \n"
        text += "x - Convolution with x-derivative \n"
        text += "y - Convolution with x-derivative \n"
        text += "m - Magnitude of gradient \n"
        text += "p - Plot the gradient vectors \n"
        text += "r - Rotate the image with custom angle \n"
        text += "h - Display Help \n"

        y0, dy = 10, 25
        for i, line in enumerate(text.split('\n')):
            y = y0 + i*dy
            cv2.putText(img, line, (50, y ), font,fontScale,fontColor,lineType)
        cv2.imshow('Capture',img)