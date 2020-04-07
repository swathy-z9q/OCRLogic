import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
import math
from datetime import datetime 
import pandas as pd
import statistics as stats

class preprocessing:

    def rotate_bound(self, image, angle):
        
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2) #centre of the image
        
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH),borderValue=(255,255,255))

    def getOrientationInfo(self, image):
        imageInfo = {}
        isBlankPage = False
        
        doc = image.copy()    
        # smooth the image to avoid noises
        imageBlurred = cv2.medianBlur(image,5)
        
        thresh = cv2.adaptiveThreshold(imageBlurred,255,1,1,11,2)
        thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

        # dilation and erosion to join the gaps - change iteration to detect more or less area's
        dilated = cv2.dilate(thresh_color,None,iterations = 5)
        eroded = cv2.erode(dilated,None,iterations = 10)

        kernel = np.ones((1,30), np.uint8) 
        d_im = cv2.dilate(eroded, kernel, iterations=1)
        e_im = cv2.erode(d_im, kernel, iterations=1) 

        gray = cv2.cvtColor(e_im,cv2.COLOR_BGR2GRAY) 
        cnts, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        angles = []
        widths = []
        heights = []
        noOfCnts = 0
        for i,c in enumerate(cnts):      
            area = cv2.contourArea(c)
            if(area>500):
                noOfCnts += 1
                rect = cv2.minAreaRect(c)
                angles.append(rect[-1])
                box = cv2.boxPoints(rect) 
                box = np.int0(box)
                for i in range(len(box)):
                    if(int(abs(rect[-1])) == 0 and abs((box[0]- box[1])[1]) > abs((box[0]- box[3])[0])):
                        heights.append(abs((box[0]- box[1])[1]))
                        widths.append(abs((box[0]- box[3])[0]))

        for i,a in enumerate(angles):
            if(abs(a) < 5 or 85<= abs(a) <= 90):
                angles[i] = 0.0         
        if(len(angles) == 0): #In case of blank page with or wihtout noise
            angle = 0
            isBlankPage = True
            isSkewed = False
            isLayoutMode = False
        
        else:
            angle = np.median(angles) 
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            if(len(heights) > 0 and len(widths) > 0):   #To check whether page is horizontally oriented 
                isLayoutMode = True if (np.mean(heights) > np.mean(widths) and len(heights)/noOfCnts > 0.60) else False
            else:
                isLayoutMode = False
        isSkewed = False if int(abs(angle)) == 0 else True
        imageInfo = {'Layout_mode': isLayoutMode, 'Blank_page':isBlankPage, 'isSkewed': isSkewed, 'Skew_angle': angle}
        return imageInfo

    def orientImage(self, doc, isPotraitMode = False):
        
        docInfo = self.getOrientationInfo(doc)
        
        isLayoutMode = docInfo['Layout_mode']
        angle = docInfo['Skew_angle']
        if( (int(angle) == 0 or (abs(angle) < 10)) and isLayoutMode):
            doc = self.rotate_bound(doc,-90)
            return doc

        if(isPotraitMode):
            return doc
            
        if(round(abs(angle)) <= 5 and not isLayoutMode): #Current page is in potrait mode hence deskewing not needed
            return doc

        elif(abs(angle)>5):
            doc = self.rotate_bound(doc, angle)
            docInfo = self.getOrientationInfo(doc)
            return self.orientImage(doc, True)
        else:
            pass   
        return doc
        

    def removeSmallDotNoiseAlternate(self, image, debug, debugImgPath):
        ret,image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, None, None, None, 8, cv2.CV_32S)
        sizes = stats[1:, -1] #get CC_STAT_AREA component
        img2 = np.zeros((labels.shape), np.uint8)
        
        print(nlabels)
        for i in range(0, nlabels - 1):
            if sizes[i] >= 20:   #filter small dotted regions
                img2[labels == i + 1] = 255

        image = cv2.bitwise_not(img2)

        ret = None
        img2 = None
        sizes = None
        nlabels = None
        labels = None
        stats = None
        centroids = None

        if debug:
            cv2.imwrite(debugImgPath, image)

        return image
    

    def removeBg(self,contrastedImage,image_name):
        isBgRemoved = False
        gray = contrastedImage
        img3 = gray.copy()

        #gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        image = np.stack((gray,)*3, axis=-1)

        blurred_image = cv2.GaussianBlur(image,(5,5),0)

        a = blurred_image.max() 
        _, thresh = cv2.threshold(blurred_image, a/2+60, a,cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(thresh,cv2.COLOR_BGR2HSV)

        #dst = cv2.fastNlMeansDenoising(mask,None,10,7,21)
        high_black = [0, 0, 0]
        high_black = np.array(high_black, dtype="uint8")

        mask = cv2.inRange(hsv,high_black,high_black)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cnts = []
        areaa = []
        for c,cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            areaa.append(area)
            if(area > 10000):
                cnts.append(cnt)
        if(len(cnts)>0):
            isBgRemoved = True    
            final = cv2.drawContours(image, cnts, contourIdx = -1, color = (255, 0, 0), thickness = -1)

            ROI_Range = [255, 0, 0]

            # create NumPy arrays from the boundaries
            ROI_Range = np.array(ROI_Range, dtype="uint8")
            mask_ROI = cv2.inRange(final, ROI_Range, ROI_Range)

            a = mask_ROI.max() 
            _, mask_ROI = cv2.threshold(mask_ROI, a/2+60, a,cv2.THRESH_BINARY)

            mask_ROI_inverted = cv2.bitwise_not(mask_ROI)


            inverted_ROI = cv2.bitwise_not(img3,mask = mask_ROI)
            # Take only region of inverted  image.
            img2_fg = cv2.bitwise_and(img3,img3,mask = mask_ROI)

            #Black + Any Color = Anycolor
            #White + Any Color = White
            # Now black-out the area of logo in ROI
            img1_bg = cv2.bitwise_and(img3,img3,mask = mask_ROI_inverted)

            #cv2.imwrite('pag.jpg',img1_bg)
            #cv2.imwrite('testfg11.jpg',img2_fg)

            dst = cv2.bitwise_or(img1_bg,inverted_ROI)

    #       cv2.imwrite(image_name,dst)
        if(isBgRemoved):
            return dst
                
        return contrastedImage


    def removeBackgroundByContour(self, image):
        edged = cv2.Canny(image, 30, 200)
        cv2.imwrite('/Users/swathi/Documents/debug/edges.jpg', edged)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if contour.size >= 1000:
                x,y,w,h = cv2.boundingRect(contour)
                if 3000 < w < 4000:
                    print((x,y,w,h))
                    cv2.putText(image, str((w,h)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
                    cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), -1)
        cv2.imwrite('/Users/swathi/Documents/debug/contour.jpg', image)


    #Try remove contouring with re  moval of small noise
    def removeHorizontalLinesAndSmallDotsByContour(self, image):
        
        edged = cv2.Canny(image, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if contour.size >= 1000:
                x,y,w,h = cv2.boundingRect(contour)
                if h < 30:
                    cv2.rectangle(image, (x,y), (x+w, y+h), (255, 255, 255), -1)
            elif contour.size < 40:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x,y), (x+w, y+h), (255, 255, 255), -1)
        return image

    def removeVerticalLinesByContour(self, image):
        edged = cv2.Canny(image, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if contour.size >= 1000:
                x,y,w,h = cv2.boundingRect(contour)
                if w < 30:
                    cv2.rectangle(image, (x,y), (x+w, y+h), (255, 255, 255), -1)
        return image

    def removeHorizontalLinesByMorphology(self, image):
        for iteration in range(0, 10):
            ## (1) Create long line kernel, and do morph-close-op
            kernel = np.ones((1,50), np.uint8)
            morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

            ## (2) Invert the morphed image, and add to the source image:
            dst = cv2.add(image, (255-morphed))

        horizontal_size = 10
        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        # Apply morphology operations
        dst = cv2.erode(dst, horizontalStructure)
        dst = cv2.dilate(dst, horizontalStructure)
        return dst
        

    def removeHorizontalLines(self, image, debug, debugImgPath):
        image = self.removeHorizontalLinesAndSmallDotsByContour(image)
        # image = removeHorizontalLinesByMorphology(image)
        if debug:
            cv2.imwrite(debugImgPath, image)
        
        return image

    def increaseContrast(self, image, debug, debugImgPath):
        contrast = 64
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
        ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        image = cv2.GaussianBlur(image,(3,3),0)

        if debug:
            cv2.imwrite(debugImgPath, image)

        return image

    def removeVerticalLines(self, image, debug, debugImgPath):
        image = self.removeVerticalLinesByContour(image)
        if debug:
            cv2.imwrite(debugImgPath, image)
        return image

    def imageEnhance(self, image, debug, debugImgPath):
        image = cv2.fastNlMeansDenoising(image, None, 40, 10, 40)
        image = cv2.detailEnhance(image)

        if debug:
            cv2.imwrite(debugImgPath, image)

        return image

    def removeIslandNoise(self, image):
        img_bw = 255*(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

        mask = np.dstack([mask, mask, mask]) / 255
        output = image * mask

        return output
