from PIL import ImageFont, ImageDraw, Image
from collections import defaultdict
import numpy as np
import pandas as pd
import uuid
import cv2
import time
import re

import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import skimage
from skimage import morphology


ALLIGNMENT_DEVIATION = 25

class imagePostProcessing:
    def isColor(self, img):
        return (img is not None) and hasattr(img[1,1], "__len__")

    def convertToColor(self, img):
        if not self.isColor(img):
            return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            # return cv2.convertScaleAbs(img, alpha=2, beta=50)
        return img

    def clearNonTable(self, sentences, image):
        for dr in sentences:
            if(dr['potentialTable'] == False):    
                (x, y, w, h) = (dr['left'], dr['top'], dr['width'], dr['height'])
                cv2.rectangle(image, (x, y-5), (x + w, y + h + 5), (255, 255, 255), -1)
        
        return image

    def clearUnwantedBlock(self, block, image, index):
        (x, y, w, h) = (block['left'][index], block['top'][index], block['width'][index], block['height'][index])
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)
        return image

    def clearUnwantedSentences(self, sentences, image):
        for dr in sentences:
            if(dr['text'] == ''):
                (x, y, w, h) = (dr['left'], dr['top'], dr['width'], dr['height'])
                cv2.rectangle(image, (x, y-5), (x + w, y + h + 5), (255, 255, 255), -1)
        return image


    def getHeaders(self, d):

        nonEmptyI = [i for i in range(len(d['text'])) if len(d['text'][i]) > 0 and d['conf'][i] >= 60]
        allCharWidth = [ int(d['width'][i]) / (len(d['text'][i])) for i in list(nonEmptyI)]
        medHeight = np.median([d['height'][i] for i in list(nonEmptyI)])
        medCharWidth = np.median(allCharWidth)
        wordInfo = pd.DataFrame(d)
        lineInfo = wordInfo['line_num'].value_counts(ascending = True).reset_index()
        
        isHeightGreater = []
        isCharWidthGreater = []
        isAlphaNum = []
        lineWordCount = [] 
    
        for idx in range(len(d['text'])):
            if idx in nonEmptyI:
                lineNo = d['line_num'][idx]
                wordCount = int(lineInfo[lineInfo['index'] == lineNo]['line_num'].values[0])
                lineWordCount.append(wordCount)
                if(medHeight < d['height'][idx]):
                    isHeightGreater.append(1)
                else:
                    isHeightGreater.append(0)
                if(medCharWidth < d['width'][idx] / (len(d['text'][idx])) ):
                    isCharWidthGreater.append(1)
                else:
                    isCharWidthGreater.append(0)

                if(d['text'][idx].isalnum()):
                    isAlphaNum.append(1)
                else:
                    isAlphaNum.append(0)
            else:
                lineWordCount.append(0)
                isHeightGreater.append(0)
                isCharWidthGreater.append(0)
                isAlphaNum.append(0)
    
        d['lineWordCount'] =  lineWordCount 
        d['isHeightGreater'] = isHeightGreater
        d['isCharWidthGreater'] =isCharWidthGreater
        d['isAlphaNum'] = isAlphaNum
        return medHeight,medCharWidth

    def getTextInfo(self, l, image, image_name, mHeight,mCharWidth ):
        image = self.convertToColor(image)
        textInfo = {}
        wordTexts = []
        wordConf = []
        height = []
        topGap = []
        pixelIntensity = []
        wordCount = []
        charWidth = []
        left = []
        top = []
        height = []
        width = []
        isHeightGreater = []
        bottomGap = []
        isnum = []
        lineNumbers = []
        pixelIntensityRatio = []
        pixelIntensity
        area = []

        thresh = self.displayOCRWords(3, image, image_name)
        for lineNo,line in enumerate(l):
            for sentNo,sent in enumerate(line['sentences']):
                for wordNo,word in enumerate(sent['words']):
                    height.append(word['height'])
                    if(len(word['text'])>0):
                        cWidth = word['width']/len(word['text'])    
                        charWidth.append(cWidth)
                    else:
                        charWidth.append(0)
                    if(word['conf'] < 30):
                        pixelIntensity.append(0)
                    else:
                        (x, y, w, h) = (word['left'], word['top'], word['width'], word['height'])
                        ROI = thresh[y:y + h, x:x + w] 
                        aa = ROI.shape[0] * ROI.shape[1]
                        area.append(len(word['text']))
                        n_white_pix = round(np.sum(ROI == thresh.max())/ (ROI.shape[0] * ROI.shape[1])) * 100
                        a = round(np.sum(ROI == thresh.max())/ (ROI.shape[0] * ROI.shape[1]) * 100)
                        pixelIntensity.append(a)
                        pixelIntensityRatio.append( a / (ROI.shape[0] * ROI.shape[1])) 
                        #cv2.putText(image,str(a), (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                        
                
        isLowIntensityVar =  False
        isHighIntensityVar = False
        if(round(np.mean(height)) >= 50 and round(np.median(charWidth)) >= 30):
            thresh = self.displayOCRWords(3, image, image_name)
        else:
            thresh = self.displayOCRWords(3, image, image_name)
        if( np.percentile(pixelIntensity,90) - np.percentile(pixelIntensity,60) < 3 ):
            thresh = self.displayOCRWords(1, image, image_name)
            isLowIntensityVar =  True
        
        pixelIntensity = []
        for lineNo,line in enumerate(l):
            for sentNo,sent in enumerate(line['sentences']):
                for wordNo,word in enumerate(sent['words']):

                    wordConf.append(word['conf']) 
                    lineNumbers.append(lineNo)
                    if(mHeight < word['height']):
                        isHeightGreater.append(1)
                    else:
                        isHeightGreater.append(0)

                    if(word['text'].isnumeric()):
                        isnum.append(1)
                    else:
                        isnum.append(0)

                    left.append(word['left'])
                    top.append(word['top'])
                    width.append(word['width'])    
                    wordTexts.append(word['text'])
                    #if(line['potentialTable'] == True):
                    #    isTable = 1
                    #lse:
                    #    isTable = 0
                    #isPotentialTable.append(isTable)
                    wordCount.append(len(sent['words']))
                    if(lineNo == 0):
                        topGap.append(line['top'])
                    else:
                        prevLineNo = lineNo -1
                        diff = line['top'] - l[prevLineNo]['top']
                        topGap.append(diff)
                    
                    if(lineNo == len(l) - 1):
                        bottomGap.append(0)
                    else:
                        nextLineNo = lineNo + 1
                        bDiff =  l[nextLineNo]['top'] - (line['top'] + line['height'])
                        bottomGap.append(bDiff)
                    if(word['conf'] < 30):
                        pixelIntensity.append(0)
                    else:
                        (x, y, w, h) = (word['left'], word['top'], word['width'], word['height'])
                        ROI = thresh[y:y + h, x:x + w] 
                        n_white_pix = np.sum(ROI == thresh.max())/ (ROI.shape[0] * ROI.shape[1])
                        n_black_pix = np.sum(ROI == thresh.min())/ (ROI.shape[0] * ROI.shape[1])
                        #cv2.putText(image,str(round(n_white_pix * 100,2)), (x - 80, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        pixelIntensity.append(round(n_white_pix * 100))
                        
        percentRank = [stats.percentileofscore(pixelIntensity,pI) for pI in pixelIntensity]
        textInfo['lineNumber'] = lineNumbers           
        textInfo['wordTexts'] = wordTexts
        wordLen = [len(word) for word in wordTexts]
        #   textInfo['isPotentialTable'] = isPotentialTable
        textInfo['height'] = height
        textInfo['top'] = top
        textInfo['wordLen'] = wordLen
        textInfo['left'] = left
        textInfo['width'] = width
        textInfo['topGap'] = topGap
        textInfo['pixelIntensity'] = pixelIntensity
        textInfo['wordCount'] = wordCount
        textInfo['charWidth'] = charWidth
        textInfo['percentRank'] = percentRank
        #textInfo['isCharWidthGreater'] = isCharWidthGreater
        textInfo['isHeightGreater'] = isHeightGreater
        textInfo['bottomGap'] = bottomGap
        textInfo['isnum'] = isnum
        medBottomGap = np.median(bottomGap)
        medTopGap = np.median(topGap)
        medPixel = np.percentile(pixelIntensity,80)
        textInfo['isTopGapOutlier'] = [1 if textInfo['topGap'][i] > medTopGap else 0 for i in range(len(textInfo['left']))]
        textInfo['isBottomGapOutlier'] = [1 if textInfo['bottomGap'][i] > medBottomGap else 0 for i in range(len(textInfo['left']))]
        textInfo['isMoreDense'] = [1 if textInfo['pixelIntensity'][i] > medPixel else 0 for i in range(len(textInfo['left']))]
        textInfo['isPotentialBold'] = [1 if ((textInfo['isTopGapOutlier'][i] or textInfo['isBottomGapOutlier'][i]) and textInfo['isMoreDense'][i]) else 0 for i in range(len(textInfo['left']))]
        data = pd.DataFrame(textInfo)
        sil_score_max = -1 #minimum possible score
        d = data.groupby('lineNumber', as_index=False).agg({"pixelIntensity": "mean"}).sort_values('pixelIntensity',ascending = True).reset_index()
        lineIntensityMean = []
        wordVarianceFromLine = []

        for i in range(len(textInfo['pixelIntensity'])):
            lNo = textInfo['lineNumber'][i]
            lineMeanIntensity = d[d['lineNumber'] == lNo]['pixelIntensity'].values[0]
            lineIntensityMean.append(lineMeanIntensity)
            wordVarianceFromLine.append(np.var([textInfo['pixelIntensity'][i],lineMeanIntensity]))

        textInfo['lineIntensityMean'] = lineIntensityMean
        textInfo['wordVarianceFromLine'] = wordVarianceFromLine
        data = pd.DataFrame(textInfo)
        maxVarIndex = data.groupby('lineNumber', as_index=False)['wordVarianceFromLine'].idxmax()
        meanP = np.median(lineIntensityMean)
        #if meanP < 2:
        #    isHighIntenseWord = np.zeros(len(textInfo['wordTexts']),dtype = int)
        #else:
        isHighIntenseWord = [1 if (i in maxVarIndex.values and textInfo['pixelIntensity'][i] > meanP) else 0 for i in range(len(textInfo['pixelIntensity']))]
        textInfo['isHighIntenseWord'] = isHighIntenseWord
        boldLines = []
        for i in np.unique(textInfo['lineNumber']):
            if  1 <= int(np.median(data[data['lineNumber'] == i]['lineIntensityMean'])) >= int(np.percentile(lineIntensityMean,95) and data['wordVarianceFromLine'][i] > 1):
                boldLines.append(i)
        for i in range(len(textInfo['wordTexts'])):
            if i in boldLines:
                textInfo['isHighIntenseWord'][i] = 1
        data = pd.DataFrame(textInfo)
        
        if(isLowIntensityVar ==  False):
            if(np.var(height) > 100 and np.percentile(pixelIntensity,90) - np.percentile(pixelIntensity,60) < 10):
                features = ['pixelIntensity','height','charWidth']
            else:
                features = ['pixelIntensity','isHighIntenseWord','lineIntensityMean','wordVarianceFromLine']
            X =  preprocessing.scale(data[features])

            for n_clusters in range(2,6):
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
                pred_y = kmeans.fit_predict(X)
                sil_score = silhouette_score(X, pred_y)
                if sil_score > sil_score_max:
                    sil_score_max = sil_score
                    labels = pred_y
                    best_n_clusters = n_clusters
        
            kmeans = KMeans(n_clusters = best_n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)               
            pred_y = kmeans.fit_predict(X)
            textInfo['label'] = pred_y
            textInfo['wordConf'] = wordConf
            data = pd.DataFrame(textInfo)
            if(np.var(height) < 100):
                groupbyRes = data.groupby('label', as_index=False).agg({"pixelIntensity": "mean"}).sort_values('pixelIntensity',ascending = True).reset_index()
                idx = groupbyRes[groupbyRes['label'] == np.median(pred_y)].index
                labelBold = groupbyRes[idx.values[0]+1:]['label'].values #groupbyRes['label'][0] #groupbyRes['label'].values[-1:]# if  best_n_clusters >=4 else groupbyRes['label'].values[-1:]
                if len(labelBold) == 0:
                    leastPixel = groupbyRes['label'][0]
                    labelBold = groupbyRes[groupbyRes['label'] != leastPixel]['label'].values
            else:
                groupbyRes = data.groupby('label', as_index=False).agg({"height": "mean"}).sort_values('height',ascending = True).reset_index()
                idx = groupbyRes[groupbyRes['label'] == np.median(pred_y)].index
                labelBold = groupbyRes[idx.values[0]+1:]['label'].values #groupbyRes['label'][0] #groupbyRes['label'].values[-1:]# if  best_n_clusters >=4 else groupbyRes['label'].values[-1:]
                if len(labelBold) == 0:
                    leastPixel = groupbyRes['label'][0]
                    labelBold = groupbyRes[groupbyRes['label'] != leastPixel]['label'].values

            for i in range(len(textInfo['wordTexts'])):
                (x, y, w, h) = (textInfo['left'][i], textInfo['top'][i], textInfo['width'][i], textInfo['height'][i])
                if (textInfo['label'][i] in labelBold):
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(image,str(data['pixelIntensity'][i]), (x - 80, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                cv2.putText(image,str(features), (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                

        else:
            p = np.array(pixelIntensity).astype(int)
            data['p'] = p
            percentRank = [stats.percentileofscore(pixelIntensity,pI) for pI in pixelIntensity]
            pIndex = data[data['p'] != 0].index
            lp = data[data['p'] != 0]['p']
            
            for i in range(len(textInfo['wordTexts'])):
                if(len(textInfo['wordTexts']) > 1):
                    (x, y, w, h) = (textInfo['left'][i], textInfo['top'][i], textInfo['width'][i], textInfo['height'][i])
                    if percentRank[i] > 80 and data['p'][i] > np.percentile(pixelIntensity,90):#(data['p'][i] > np.percentile(lp,10) or data['isHighIntenseWord'][i] == 1):
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(image,str(data['p'][i]), (x - 80, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
        data.to_csv('/users/swathi/documents/KernelCheckTrainImage/_' + image_name + '_Features_.csv')

        varI = np.percentile(pixelIntensity,90) -  np.percentile(pixelIntensity,60) 
        cv2.putText(image,'LineIntensityMean: '+ str(np.mean(lineIntensityMean)) +'Height: '+ str(np.mean(height))+'charWidth: '+ str(np.mean(charWidth)) + 'PixelVariance' + str(varI) , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
        return image
        
    
    def getStrokedImage(self, k, image, imageName):

        #if k == 'poor':
        img = cv2.medianBlur(image,5) 
        a = img.max()
        _, thresh = cv2.threshold(img, a/2+60, a,cv2.THRESH_BINARY)
        thresh_inv = cv2.bitwise_not(thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(4,4)) #np.ones((3,3),np.uint8)
        op = cv2.erode(thresh_inv,kernel,iterations = 2)

        op = cv2.cvtColor(op,cv2.COLOR_BGR2GRAY)
        binarized = np.where(op>0.1, 1, 0)
        processed = morphology.remove_small_objects(binarized.astype(bool), min_size=20, connectivity=3).astype(int)

        # black out pixels
        mask_x, mask_y = np.where(processed == 0)
        op[mask_x, mask_y] = 0
    #if k == 'good':
        #    img = cv2.medianBlur(image,5) 
        #    a = img.max()
        #    _, thresh = cv2.threshold(img, a/2+60, a,cv2.THRESH_BINARY)
        #    thresh_inv = cv2.bitwise_not(thresh)
        #    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) #np.ones((3,3),np.uint8)
        #    op = cv2.erode(thresh_inv,kernel,iterations = 2)
        #    op = cv2.cvtColor(op,cv2.COLOR_BGR2GRAY)
        #    binarized = np.where(op>0.1, 1, 0)
        #    processed = morphology.remove_small_objects(binarized.astype(bool), min_size=40, connectivity=10).astype(int)
            # black out pixels
        #    mask_x, mask_y = np.where(processed == 0)
        #    op[mask_x, mask_y] = 0
        """
            a = img.max()
            _, thresh = cv2.threshold(img, a/2+60, a,cv2.THRESH_BINARY)
            thresh_inv = cv2.bitwise_not(thresh)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(k,k)) #np.ones((3,3),np.uint8)
            op = cv2.erode(thresh_inv,kernel,iterations = 2)
            if(k == 4):
                kernel = np.ones((3,3),np.uint8)
                op = cv2.dilate(op,kernel,iterations = 2)
                
            img2d = cv2.cvtColor(op,cv2.COLOR_BGR2GRAY)
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img2d, None, None, None, 8, cv2.CV_32S)
            areas = stats[1:,cv2.CC_STAT_AREA]
            result = np.zeros((labels.shape), np.uint8)
            checkArea = np.percentile(areas,85)

            for i in range(0, nlabels - 1):
                if areas[i] > checkArea:   #keep
                    result[labels == i + 1] = 255
            a = result.max()       
        """

        #_, thresh = cv2.threshold(op, a/2+60, a,cv2.THRESH_BINARY)
        cv2.imwrite('/users/swathi/documents/KernelCheckOakTree/_' + imageName,op)
        return op

    def getBoldText(self, l, image, imageName):
        #k = 'good'
        image = self.convertToColor(image)
        #thresh = self.getStrokedImage(5,image,imageName)
        textInfo = {}
        height = []  
        wordConf = []
        charWidth = []
        wordTexts = []
        lineNumbers = []
        pixelIntensity = []
        lineIntensityMean = []
        wordVarianceFromLine = []
        left = []
        top = []
        width = []
        height = []

        #1.Identify image type for bold extraction
        for lineNo,line in enumerate(l):
            for sentNo,sent in enumerate(line['sentences']):
                for wordNo,word in enumerate(sent['words']):
                    wordTexts.append(word['text'])
                    left.append(word['left'])
                    top.append(word['top'])
                    width.append(word['width'])    
                    lineNumbers.append(lineNo)
                    height.append(word['height'])
                    if(len(word['text'])>0):
                        cWidth = word['width']/len(word['text'])    
                        charWidth.append(cWidth)
                    else:
                        charWidth.append(0)
                    
        textInfo['height'] = height
        textInfo['charWidth'] = charWidth
        textInfo['top'] = top
        textInfo['left'] = left
        textInfo['width'] = width
        #textInfo['pixelIntensity'] = pixelIntensity
        textInfo['wordTexts'] = wordTexts
        
        textInfo['lineNumber'] = lineNumbers

        if( 45 <= round(np.median(height))  and round(np.median(charWidth)) >= 21 ):
            k = 'good'
        else:
            k = 'poor'
        pixelIntensity = []
        thresh = self.getStrokedImage(k,image,imageName)
        for lineNo,line in enumerate(l):
            for sentNo,sent in enumerate(line['sentences']):
                for wordNo,word in enumerate(sent['words']):
                
                    (x, y, w, h) = (word['left'], word['top'], word['width'], word['height'])
                    ROI = thresh[y:y + h, x:x + w] 
                    n_white_pix = np.sum(ROI == thresh.max())/ (ROI.shape[0] * ROI.shape[1])
                    #cv2.putText(image,str(round(n_white_pix * 100,2)), (x - 80, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    pixelIntensity.append(round(n_white_pix * 100))
        textInfo['pixelIntensity'] = pixelIntensity
        percentRank = [stats.percentileofscore(pixelIntensity,pI) for pI in pixelIntensity]
        #data = pd.DataFrame(textInfo)
        #d = data.groupby('lineNumber', as_index=False).agg({"pixelIntensity": "mean"}).sort_values('pixelIntensity',ascending = True).reset_index()
        #lineIntensityMean = []
        #wordVarianceFromLine = []
        #for i in range(len(textInfo['pixelIntensity'])):
        #    lNo = textInfo['lineNumber'][i]
        #    lineMeanIntensity = d[d['lineNumber'] == lNo]['pixelIntensity'].values[0]
        #    lineIntensityMean.append(lineMeanIntensity)
        #   wordVarianceFromLine.append(np.var([textInfo['pixelIntensity'][i],lineMeanIntensity]))
        #textInfo['lineIntensityMean'] = lineIntensityMean
        #textInfo['wordVarianceFromLine'] = wordVarianceFromLine
        textInfo['percentRank'] = percentRank
        data = pd.DataFrame(textInfo)
        #if(k == 5):    
        #    for i in range(len(textInfo['wordTexts'])):
        #        if(len(textInfo['wordTexts']) > 1):
        #           (x, y, w, h) = (textInfo['left'][i], textInfo['top'][i], textInfo['width'][i], textInfo['height'][i])  
        #            if percentRank[i] >= 60: #and data['p'][i] > np.percentile(pixelIntensity,90):#(data['p'][i] > np.percentile(lp,10) or data['isHighIntenseWord'][i] == 1):
        #                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #            cv2.putText(image,str(round(percentRank[i],2)), (x - 80, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        
        #if(k == 4 or k==3):
        X =  preprocessing.scale(data[['pixelIntensity','percentRank','width']])
        sil_score_max = -1
        for n_clusters in range(2,6):
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
            pred_y = kmeans.fit_predict(X)
            sil_score = silhouette_score(X, pred_y)
            if sil_score > sil_score_max:
                sil_score_max = sil_score
                labels = pred_y
                best_n_clusters = n_clusters
        kmeans = KMeans(n_clusters = best_n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)               
        pred_y = kmeans.fit_predict(X)
        data['pred'] = pred_y
        data.groupby('pred', as_index=False).agg({"pixelIntensity": "mean"}).sort_values('pixelIntensity',ascending = True).reset_index()
        groupbyRes = data.groupby('pred', as_index=False).agg({"pixelIntensity": "mean"}).sort_values('pixelIntensity',ascending = True).reset_index()
        idx = groupbyRes[groupbyRes['pred'] == np.median(pred_y)].index
        labelBold = groupbyRes[idx.values[0]+1:]['pred'].values
        for i in range(len(textInfo['wordTexts'])):
            (x, y, w, h) = (textInfo['left'][i], textInfo['top'][i], textInfo['width'][i], textInfo['height'][i])
            if (data['pixelIntensity'][i] > 2):
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(image,str(round(data['pred'][i],2)), (x - 80, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
        #pTotal = [i for i in pixelIntensity if i > 5]
        #checkP = np.percentile(percentRank,85)
        #for i in range(len(textInfo['wordTexts'])):
        #    if(len(textInfo['wordTexts']) > 1):
        #        (x, y, w, h) = (textInfo['left'][i], textInfo['top'][i], textInfo['width'][i], textInfo['height'][i])  
        #        if percentRank[i] >= checkP or percentRank[i] >= 70:
        #           cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #        cv2.putText(image,str(round(percentRank[i],2)), (x - 80, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        data.to_csv('/users/swathi/documents/KernelCheckOakTree/_' + imageName + '_Features_.csv')
        return image


    def displayOCRWords(self, kernelSize, image, image_name):

        #kValue = self.getKernelSize(textInfo)
        #print(image_name,kValue)

        img = cv2.medianBlur(image,5) 
        a = img.max()
        _, thresh = cv2.threshold(img, a/2+60, a,cv2.THRESH_BINARY)
        thresh_inv = cv2.bitwise_not(thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) #np.ones((3,3),np.uint8)
        erosion = cv2.erode(thresh_inv,kernel,iterations = 2)
        
        #kernel = np.ones((3,3),np.uint8)
        #opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
        #a = opening.max()
        _, thresh = cv2.threshold(opening, a/2+60, a,cv2.THRESH_BINARY)
        cv2.imwrite('/users/swathi/documents/KernelCheckOakTree/_' + image_name,thresh)
        
        return cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)


    def CrossdisplayOCRWords(self, kernelSize, image, image_name):

        debugImage = self.convertToColor(image)
        #n_boxes = len(d['level'])
        #wImage = image.copy()
        #wImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(image,5) 
        a = img.max() 
        _, thresh = cv2.threshold(img, a/2+60, a,cv2.THRESH_BINARY)
        thresh_inv = cv2.bitwise_not(thresh)
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(thresh_inv,kernel,iterations = 2)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((3,3),np.uint8)
        dilation = cv2.dilate(opening,kernel,iterations = 3)
        kernel = np.ones((kernelSize,kernelSize),np.uint8)
        final = cv2.erode(dilation,kernel,iterations =kernelSize)
        a = final.max()
        _, thresh = cv2.threshold(final, a/2+60, a,cv2.THRESH_BINARY)
        cv2.imwrite('/users/swathi/documents/kernelcheck/_' + image_name,final)
        return cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
    
        """
            pixelIntensity = []
            for i in range(n_boxes):
                if(d['level'][i] == 5):
                    if(d['height'][i] != d['width'][i]):
                        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                        ROI = thresh[y:y + h, x:x + w] 
                        n_white_pix = np.sum(ROI == thresh.max())/ (ROI.shape[0] * ROI.shape[1])
                        #cv2.putText(image,str(round(n_white_pix * 100,2)), (x - 80, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        pixelIntensity.append(n_white_pix * 100)
                    else:
                        pixelIntensity.append(0)   
                else:
                    pixelIntensity.append(0)
            
            nonEmptyI = [i for i in range(len(d['text'])) if len(d['text'][i]) > 0]
            allCharWidth = []
            for i in range(len(d['text'])):
                if i in nonEmptyI:
                    allCharWidth.append(int(d['width'][i] - ((len(d['text'][i]) - 1 )* 7)) / (len(d['text'][i])))
                else:
                    allCharWidth.append(0)

            #allCharWidth = [ int(d['width'][i] - ((len(d['text'][i]) - 1 )* 7)) / (len(d['text'][i])) for i in ]
            d['Pixelintensity'] = pixelIntensity

            #d['charWidth'] = allCharWidth
            
                if(np.sum(pixelIntensity) > 10):
                    nonEmptyI = [i for i in range(len(d['text'])) if len(d['text'][i]) > 0]
                    medHeight = np.median([d['height'][i] for i in list(nonEmptyI)])
                    result = pd.DataFrame(d)
                    X =  preprocessing.scale(result[['Pixelintensity','charWidth']])

                    sil_score_max = -1 #minimum possible score
                    
                    for n_clusters in range(2,6):
                        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
                        pred_y = kmeans.fit_predict(X)
                        sil_score = silhouette_score(X, pred_y)
                    #print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
                        if sil_score > sil_score_max:
                            sil_score_max = sil_score
                            best_n_clusters = n_clusters

                    kmeans = KMeans(n_clusters = best_n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
                    ##if(best_n_clusters > 2):
                    ##  print("hi")
                    #    pass
                    pred_y = kmeans.fit_predict(X)
                    result['pred'] = pred_y
                    groupRes = result.groupby('pred', as_index=False).agg({"Pixelintensity": "mean"}).sort_values('Pixelintensity',ascending = True).reset_index()
                    #threshIntensity = np.median(groupRes['Pixelintensity']) if np.percentile(pixelIntensity,70) < 3 else np.percentile(pixelIntensity,70)#np.median(groupRes['Pixelintensity']) - 2
                    #groupRes = groupRes[groupRes['Pixelintensity'] >= threshIntensity] 
                    f = groupRes['pred'].values[-1:][0] #groupRes['pred'].values # 
                    #If Pixel intensity over the page greater than 5% of intensi  
                    if(np.sum(groupRes['Pixelintensity']) > 5):
                        d = result
                        for i in range(n_boxes):
                            if(d['level'][i] == 5):
                                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                                ROI = thresh[y:y + h, x:x + w] 
                                n_white_pix = np.sum(ROI == thresh.max())/ (ROI.shape[0] * ROI.shape[1])
                                if(d['pred'][i] ==  f): #or ((d['height'][i] - medHeight) / d['height'][i]) * 100 > 35 ):
                                    cv2.rectangle(debugImage, (x, y), (x + w, y + h), (255, 0, 0), 3)
                                cv2.putText(debugImage,str(d['pred'][i]), (x - 80, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
                    r = pd.DataFrame(groupRes)
                    cv2.putText(debugImage,'PredLabels:' + str(np.array(groupRes['pred'])) + 'PixelIntensity:'+ str(np.array(groupRes['Pixelintensity'])) , (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3)
                    
                fig = plt.figure()
                plt.scatter(X[:,0], X[:,1])
                plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')
                plt.title(image_name)
                plt.xlabel('Pixel Intensity')
                plt.ylabel('Char Width')
                plt.savefig('/Users/swathi/documents/BoldInvesco_Asset_Management_Limited/'+image_name+'.png')
            
            return debugImage,d
        """
    def displayImageLines(self, lines, image):
        for dr in lines:
            (x, y, w, h) = (dr['left'], dr['top'], dr['width'], dr['height'])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(image,str(dr['tableId']), (x - 50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
            cv2.putText(image,str(dr['top']), (x + w + 50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(image,str(dr['height']), (x + w + 250, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        return image

    def drawImageSentences(self, sentences, image):
        image = self.convertToColor(image)

        fontpath = "/System/Library/Fonts/Helvetica.ttc"
        font = ImageFont.truetype(fontpath, 50)
        fontSmall = ImageFont.truetype(fontpath, 40)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        
        for dr in sentences:
            (x, y, w, h) = (dr['left'], dr['top'], dr['width'], dr['height'])
            if(len(dr['columnNo'])> 0):
                draw.text((x, y), dr['text'], font = font, fill='red')
                draw.text((x - 40, y - 20), str(dr['columnNo']), font = fontSmall, fill='green')
            else:
                draw.text((x, y), dr['text'], font = font, fill='black')
        image = np.array(img_pil)
        return image

    def displayImageSentences(self, sentences, image):
        image = self.convertToColor(image)

        for dr in sentences:
            if(dr['cLineNo'] != None):
                (x, y, w, h) = (dr['left'], dr['top'], dr['width'], dr['height'])
                # cv2.rectangle(image, (x, y-5), (x + w, y + h + 5), (255, 255, 255), -1)
                product = dr['columnNo']
                if dr['text'] == '.':
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
                else:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(image,getShape(dr['text']), (x - 50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
                # print(dr['text'])
                # cv2.putText(image,str(dr['width']), (x - 40, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
                # cv2.putText(image,str(dr['potentialTable']), (x + w + 50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
                # cv2.putText(image,str(dr['height']), (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
                cv2.putText(image,str(product), (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)

                pageNo = dr.get('isPageNo')
                if pageNo is not None and pageNo:
                    cv2.putText(image,'page no', (x - 100, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 5)
                # cv2.putText(image,str((x, y)), (x - 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
                # cv2.putText(image,str((x + w, y + h)), (x + w + 10, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
                # for wrd in dr['words']:
                #     (x, y, w, h) = (wrd['left'], wrd['top'], wrd['width'], wrd['height'])
                #     cv2.rectangle(image, (x, y), (x + w, y + h), (product, 255, 0), 1)
        return image

    
    def drawTable(self, lines, image, addText=False):
        image = self.convertToColor(image)

        tables = {}
        columns = defaultdict(dict)
        padding = 5

        for line in lines:
            if line['potentialTable'] == True:
                tableId = line['tableId']
                lineNo = line['lineNo']
                if tableId in tables:
                    tables[tableId].append(line)
                else:
                    tables[tableId] = [line]

                # Loop all the sentences and add it to columns
                for sentence in line['sentences']:
                    if len(sentence['columnNo']) > 0:
                        colNo = sentence['columnNo'][0]
                        if tableId in columns:
                            if colNo in columns[tableId]:
                                columns[tableId][colNo].append(sentence)
                            else:
                                columns[tableId][colNo] = [sentence]
                        else:
                            columns[tableId][colNo] = [sentence]

        for table in tables:
            leftX = 0
            rightX = 0
            topY = 0
            bottomY = 0
            firstRow = True
            lastLine = None
            yPoints = {}

            yIndex = 1
            for line in sorted(tables[table], key=lambda i: i['lineNo']):
                left = line['left']
                if firstRow or left < leftX:
                    leftX = left
                if firstRow:
                    topY = line['top'] - padding
                    yPoints[0] = topY
                    firstRow = False
                bottom = line['top'] + line['height'] + padding
                yPoints[yIndex] = bottom
                right = left + line['width'] + padding
                if right > rightX:
                    rightX = right
                lastLine = line
                yIndex += 1
            bottomY = lastLine['top'] + lastLine['height'] + padding
            if len(columns) > 0:
                currentTableColumns = [v for k, v in columns.items() if k == table][0]

                leftX = leftX - padding
                xPoints = {0: leftX}

                xIndex = 1
                for key in sorted(currentTableColumns.keys(), key=lambda i: i):
                    cols = currentTableColumns[key]
                    aray = np.asarray([col['right']
                                    for col in cols if len(col['columnNo']) == 1])
                    if len(aray) == 0:
                        aray = np.asarray(
                            [col['right'] for col in cols if (xIndex in col['columnNo'])])
                    if len(aray) > 0:
                        xPoints[xIndex] = int(np.max(aray)) + padding
                        xIndex += 1

                # Loop the lines and if a line does not have any sentence within a column, then create dummy sentences against that column
                for line in sorted(tables[table], key=lambda i: i['lineNo']):
                    colsInCurrentLine = []
                    # Get all columns from the sentences in the line
                    for sentence in line['sentences']:
                        if(len(sentence['columnNo']) > 0):
                            for colNo in sentence['columnNo']:
                                if colNo not in colsInCurrentLine:
                                    colsInCurrentLine.append(colNo)

                    # Loop the columns in the table and check if particular column is in table if not create dummy sentence
                    for key in sorted(currentTableColumns.keys(), key=lambda i: i):
                        try:
                            if key not in colsInCurrentLine:
                                if key in xPoints and key - 1 in xPoints:
                                    col = currentTableColumns[key]
                                    dummyWord = {'text': '', 'top': line['top'], 'left': xPoints[key - 1], 'width': xPoints[key] -
                                                xPoints[key - 1], 'height': line['height'], 'conf': 100, 'lineNum': line['lineNo'], 'id': str(uuid.uuid1())}
                                    newSentence = {'words': [dummyWord], 'top': line['top'], 'text': '', 'left': xPoints[key - 1], 'width': xPoints[key] - xPoints[key - 1], 'right': xPoints[key],
                                                'bottom': line['top'] + line['height'], 'height': line['height'], 'cLineNo': line['lineNo'], 'potentialTable': True, 'columnNo': [key], 'bulleted': False, 'id': str(uuid.uuid1())}
                                    line['sentences'].append(newSentence)
                        except Exception as err:
                            print(err)
                            pass

            lineIndex = 1
            for line in sorted(tables[table], key=lambda i: i['lineNo']):
                for sentence in line['sentences']:
                    try:
                        # Get tl, tr, bl, br for the sentence from the x and y points
                        # for each column array previous column (first index of array - 1) x point becomes left and current column x becomes right
                        # for each column (line no) previos line y becomes top and current becomes bottom
                        if(len(sentence['columnNo']) > 0):
                            allColumnNos = sorted(
                                sentence['columnNo'], key=lambda i: i)
                            currentColNo = allColumnNos[0] - 1
                            lastColNo = allColumnNos[len(allColumnNos) - 1]
                            currentLineNo = lineIndex - 1
                            tl = (xPoints[currentColNo],
                                  yPoints[currentLineNo])
                            tr = (xPoints[lastColNo], yPoints[currentLineNo])
                            bl = (xPoints[currentColNo], yPoints[lineIndex])
                            br = (xPoints[lastColNo], yPoints[lineIndex])

                            # if addText:
                            #     cv2.putText(image, sentence['text'], bl, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 1)
                            cv2.rectangle(image, tl, br, (0, 0, 255), 2)
                    except:
                        pass
                lineIndex += 1

            # Boundary
            cv2.rectangle(image, (leftX, topY),
                          (rightX, bottomY), (0, 0, 255), 2)
        return image
"""
    #Modification Start
    def getColSpan(self,ncols,sentence,LineSentence,sentNo,tableInfo,isMiddleLogic):
        leftColumnInfo,rightColumnInfo,middleColumnInfo,allignmentInfo,leftMedianInfo,rightMedianInfo = self.extractTableInfo(tableInfo)
        minDiff =10000
        colSpan = []    

        for i in range(1,ncols+1):
            for j in range(i+1,ncols+1):
                if(sentence['left'] >= leftColumnInfo[i] and sentence['right'] <=leftColumnInfo[j+1]):
                    #Check colspan with left colunmlogic
                    moreThanOneSent = 0 
                    isMiddle = 0

                    #If previous sentence Allocated with PossiblecolumnNo
                    if(sentNo>0):
                        if( self.intersection(LineSentence[sentNo - 1]['possibleColumn'], [col for col in range(i+1,j+1)])):
                            moreThanOneSent = 1

                    #If next Sentence lies within the column range which is being checked
                    if(sentNo+1 < len(LineSentence)):
                        nextSentence = LineSentence[sentNo + 1]
                        for col in range(i,j+1):
                            if (nextSentence['left'] >= leftColumnInfo[col] and nextSentence['right'] <= rightColumnInfo[col]):
                                moreThanOneSent = 1

                    #[If current sentence alone in col range] and [If sentence only within col range(i,j)]
                                
                    if isMiddleLogic :
                        columnLeftCheckLevel = middleColumnInfo[i]
                        columnRigthCheckLevel = middleColumnInfo[j]
                    else:
                        columnLeftCheckLevel = leftColumnInfo[i]
                        columnRigthCheckLevel = rightColumnInfo[j]

                    if(moreThanOneSent != 1):           
                        leftWidth  =  sentence['left'] - columnLeftCheckLevel
                        rightWidth =  columnRigthCheckLevel - sentence['right']
                        isMiddle   = 1 if (leftWidth > 0 and rightWidth > 0) else 0
                    #Potential Column Span heading 
                        if(isMiddle):
                            centreAllignDiff = abs(rightWidth - leftWidth)
                        
                            if(minDiff > centreAllignDiff):
                                minDiff  = centreAllignDiff
                                if(isMiddleLogic):
                                    colSpan = [i for i in range(i+1,j+1)]
                                else:
                                    colSpan = [i for i in range(i,j+1)]
        return colSpan,minDiff

    def extractTableInfo(self,tableInfo):
        leftColumnInfo = tableInfo['left']
        rightColumnInfo = tableInfo['right']
        middleColumnInfo = tableInfo['middle']
        allignmentInfo = tableInfo['allignment']
        leftMedianInfo = tableInfo['leftMedian']
        rightMedianInfo = tableInfo['rightMedian']
        return leftColumnInfo,rightColumnInfo,middleColumnInfo,allignmentInfo,leftMedianInfo,rightMedianInfo

    def checkIsNumber(self,text):
        number = ''.join(e for e in text if e.isalnum())
        return number.isnumeric()

    def getAllignment(self,left,right):
        lM = np.median(left)
        rM = np.median(right)
        rightAllignment = len([r for r in right if abs(rM - r) <= ALLIGNMENT_DEVIATION] ) /len(right)
        leftAllignment = len([l for l in left if abs(lM - l) <= ALLIGNMENT_DEVIATION] ) / len(left)
        if(rightAllignment>leftAllignment):
            return 'R'
        else:
            return 'L'
    
    def isAlligned(self,sentence,tableInfo,checkAllignment = 'none'):

        right = sentence['right']
        left = sentence['left']
        leftColumnInfo,rightColumnInfo,middleColumnInfo,allignmentInfo,leftMedianInfo,rightMedianInfo = self.extractTableInfo(tableInfo)
        if(checkAllignment == 'none'):
            if len(sentence['columnNo']) == 1:
                col = sentence['columnNo'][0]    
                if(allignmentInfo[col] == 'R' and not abs(rightMedianInfo[col] - right) <= ALLIGNMENT_DEVIATION):
                    return False
                elif(allignmentInfo[col] == 'L' and not abs(left - leftMedianInfo[col]) <= ALLIGNMENT_DEVIATION):
                    return False
                #elif(abs(rightColumnInfo[col] - right) <= 15 and abs(left - leftColumnInfo[col]) <= 15):
                #    return False
                else:
                    return True
            else:
                return False  
        else:
            if len(sentence['columnNo']) == 1:
                col = sentence['columnNo'][0] 
                if(checkAllignment == 'L'):
                    if(not abs(left - leftMedianInfo[col]) <= ALLIGNMENT_DEVIATION):
                        return False
                    else:
                        return True
                else:
                    if(not abs(right - rightMedianInfo[col]) <= ALLIGNMENT_DEVIATION):
                        return False
                    else:
                        return True


    #Function to find commo elements between two lists
    def intersection(cself,col1, col2):
        overlappingColumns = [] 
        overlappingColumns = [value for value in col1 if value in col2] 
        try:
            isOverLapping = True if(len(overlappingColumns)>0) else False
            return isOverLapping
        except:
            return False

    def checkIsNumber(self,text):
        number = ''.join(e for e in text if e.isalnum())
        return number.isnumeric()

    def getColumnNo(self,sentence,tableInfo):
        leftColumnInfo,rightColumnInfo,middleColumnInfo,allignmentInfo,leftMedianInfo,rightMedianInfo = self.extractTableInfo(tableInfo)
        ncols = len(allignmentInfo)
        columnNo = [0]

        for i in range(1,ncols+1):
            if (sentence['left'] >= leftColumnInfo[i] and sentence['right'] <= leftColumnInfo[i+1]):
                columnNo = [i]
        return columnNo
    

    def identifyPotentialColSpan(self,tid,ncols,tableInfo,LineSentences,Bx,Lx,image):
        print("############  Table ID :",tid)  
        #leftColumnInfo[ncols+1] = Bx  
        for (LineNo,LineSentence) in enumerate(LineSentences):
            for (sentNo,sentence) in enumerate(LineSentence):
                isColSpan = False
                colSpanMidLogic,minDiffMid = self.getColSpan(ncols, sentence, LineSentence, sentNo, tableInfo, True)
                colSpanLeftLogic,minDiffLeft = self.getColSpan(ncols, sentence, LineSentence, sentNo, tableInfo, False)
                colSpan = colSpanMidLogic if minDiffMid < minDiffLeft else colSpanLeftLogic
                if(len(colSpan)>1):
                    isColSpan = True
                #Updating column Numbers only for colSpans 
                sentence['possibleColumn'] = sentence['columnNo']
                if(isColSpan):    
                    #1. Check possible Column Span and not alligned if True then assign column Span numbers
                    if( len(colSpan) > 1 and not self.isAlligned(sentence,tableInfo)):
                        #colSpan = getColSpan(ncols, sentence, LineSentence, sentNo, tableInfo, False)
                        sentence['possibleColumn'] = colSpan
                        isColSpan = True
                    #2. if (1)  Failed and if sentence is number and is not alligned then column span numbers are assigned
                    elif(self.checkIsNumber(sentence['text']) and not self.isAlligned(sentence,tableInfo)):
                        #colSpan = getColSpan(ncols, sentence, LineSentence, sentNo, tableInfo, False)
                        #cv2.putText(image, str(colSpan), (sentence['left'], sentence['top']), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)
                        sentence['possibleColumn'] = colSpan
                        isColSpan = True
                        
                    elif(len(colSpan) > 1):
                        if(sentNo > 0):
                            if(not self.intersection(colSpan,LineSentence[sentNo-1]['possibleColumn'])):
                                sentence['possibleColumn'] = colSpan #list(set(getColSpan(ncols, sentence, LineSentence, sentNo, tableInfo, True) + getColSpan(ncols, sentence, LineSentence, sentNo, tableInfo, False)))
                                isColSpan = True
                        
                    else:
                        isColSpan = False
                
                if(not isColSpan):
                    if((self.checkIsNumber(sentence['text']) and self.isAlligned(sentence,tableInfo,'R')) or (self.getShape(sentence['text']).lower() == 'x' and self.isAlligned(sentence,tableInfo,'L'))):
                        sentence['possibleColumn'] = sentence['columnNo']
                    #3. If Tesseract column No is null or empty assign based on sentence coordinates
                    elif(len(sentence['columnNo'])==0):
                        sentence['possibleColumn'] = self.getColumnNo(sentence,tableInfo)       
                    #4. If sentence is not a colspan then assign column Number assigned form tesseract ocr
                    else:
                        sentence['possibleColumn'] = sentence['columnNo']
                
                if((self.checkIsNumber(sentence['text']) and self.isAlligned(sentence,tableInfo,'R')) or (self.getShape(sentence['text']).lower() == 'x' and self.isAlligned(sentence,tableInfo,'L'))):
                        sentence['possibleColumn'] = sentence['columnNo']       
                #Check for incorrect span:
                if(len(sentence['possibleColumn']) > 1):
                    pass
                    #if(not recheckColSpan(sentence,tableInfo)):
                    #    sentence['possibleColumn'] = sentence['columnNo']
                #Final Check if the possibleColumn and columnNo is still null then find the colspan
                if(self.isAlligned(sentence,tableInfo) and isColSpan):
                    sentence['possibleColumn'] = sentence['columnNo']
                #If a text is right alligned also check for more than one sentence
                if(self.getShape(sentence['text']).lower() == 'x' and self.isAlligned(sentence,tableInfo,'R')):
                    if(sentNo>0):
                        if( not self.intersection(LineSentence[sentNo-1]['possibleColumn'],colSpan)):
                            sentence['possibleColumn'] = colSpan
                    else:
                        sentence['possibleColumn'] = colSpan


                
                if(len(sentence['possibleColumn']) == 0):    
                    sentence['possibleColumn'] = self.getColumnNo(sentence,tableInfo) if(len(sentence['columnNo']) == 0) else sentence['columnNo']
                if(sentNo > 0):
                    if(self.intersection(sentence['possibleColumn'],LineSentence[sentNo-1]['possibleColumn'])):
                        sentence['possibleColumn'] = sentence['columnNo'] if len(sentence['columnNo']) >0 else self.getColumnNo(sentence,tableInfo)
                        cv2.putText(image, str(sentence['possibleColumn']), (sentence['left'], sentence['top']), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)
                    else:
                        cv2.putText(image, str(sentence['possibleColumn']), (sentence['left'], sentence['top']), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)
                else:
                    cv2.putText(image, str(sentence['possibleColumn']), (sentence['left'], sentence['top']), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4)
                
        return LineSentences

    def drawTableBoundary(self,lines,image,image_name):
        
        start_time = time.time()
        #Filtering based on Table id 
        TableLines = []
        #image = cv2.imread(page_path)
        TableLines = [line for line in lines if line['potentialTable']==True] 
        tableIds = set([line['tableId'] for line in TableLines])
        tableLeft = [line['left'] for line in TableLines]
        
        for tid in tableIds:
            tableGroupedById = [tline for tline in TableLines if tline['tableId'] == tid]
            
            #Find minimum Left of each column
            LineSentences = [sent['sentences'] for sent in tableGroupedById]

            #Get Height of each sentences:
            #print("Printing page with respective heights")
            #image = getHeight(LineSentences,image)
            #print("Page with heght printed")

            segments = []
            for lineSegments in LineSentences:
                segments.append(len(lineSegments))
            ncols = max(segments)
            #HashMap for Column wise average Left
            leftColumnInfo = {}
            rightColumnInfo = {}
            middleColumnInfo = {}
            allignmentInfo = {}
            leftMedianInfo = {}
            rightMedianInfo = {}
            #Iterating Column wise to find average left for each column
            for col in range(1,ncols+1):
                colLeft = []
                colRight = []
                for colSent in LineSentences:
                    for (colNo,columnGap) in enumerate(colSent,1):
                        if(len(columnGap['columnNo'])>0): #Taking left only for sentences that has column Nos or Number
                            if(len(columnGap['columnNo'])==1):
                                if columnGap['columnNo'][0] == col:
                                    #if(col == 2 or col == 4):
                                        #cv2.putText(image,str(columnGap['left'])+"--"+ str(columnGap['right']),(columnGap['left']-250,columnGap['top']), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
                                        #print("Column No:",col,"Left-->",str(columnGap['left']),str(columnGap['right']))
                                    colLeft.append(columnGap['left'])
                                    colRight.append(columnGap['right'])
                
                leftColumnInfo[col] = int(min(colLeft))   #Minimum for Left
                rightColumnInfo[col] = int(np.max(colRight)) #Maximum for Right
                allignmentInfo[col] = self.getAllignment(colLeft,colRight)
                leftMedianInfo[col] = np.median(colLeft)
                rightMedianInfo[col] = np.median(colRight)
            
            print("Table Id",tid)
            print("Allignment Information",allignmentInfo)
            
            #Table grouped by Id wise
            bottom = tableGroupedById[len(tableGroupedById)-1]
            Ly = tableGroupedById[0]['sentences'][0]['top']
            Lx = min(tableLeft)
            last_sentence_idx = len(bottom['sentences'])-1
            By = bottom['sentences'][last_sentence_idx]['bottom']
            Bx = rightColumnInfo[len(rightColumnInfo)]
            cv2.rectangle(image, (Lx,Ly),(Bx,By), (0, 255, 0), 2)

            #Finding Mid X values between columns
            for col in range(1,ncols):
                #Mx = int ((leftColumnInfo[col+1] - rightColumnInfo[col]) / 2) + rightColumnInfo[col] 
                Mx = int ((leftColumnInfo[col+1] + rightColumnInfo[col]) / 2)  
                middleColumnInfo[col] = Mx
                
            middleColumnInfo[col+1] = Bx
            leftColumnInfo[ncols+1] = Bx  
            #Constructing Table Information
            tableInfo = {}
            tableInfo['left'] = leftColumnInfo
            tableInfo['right'] = rightColumnInfo
            tableInfo['middle'] = middleColumnInfo
            tableInfo['allignment'] = allignmentInfo
            tableInfo['leftMedian'] = leftMedianInfo
            tableInfo['rightMedian'] = rightMedianInfo

            LineSentences = self.identifyPotentialColSpan(tid,ncols,tableInfo,LineSentences,Bx,Lx,image)      
             #       LineSentences = updateColSpan(tid,ncols,tableInfo,LineSentences,Bx,Lx,image)
            #Iterating Line wise to draw boundary boxes
            nextColumnLeft = 0
            curentColumnLeft = 0
            for col in range(1,ncols + 1):
                for (index,columnSentences) in enumerate(LineSentences):
                    for (sentIndex,columnLeftGap) in enumerate(columnSentences):
                        if(len(columnLeftGap['possibleColumn'])>0): #Taking only for sentences that has column Nos or Number
                            
                            if columnLeftGap['possibleColumn'][0] == col:   
                                nextColumnLeft = (leftColumnInfo[col+1] if (col != ncols) else Bx)
                                currentColumnLeft = (Lx if col == 1 else leftColumnInfo[col])
                                nextLineTop = (By if(index+1 == len(LineSentences)) else LineSentences[index+1][0]['top'])
                                
                                if(col == 1): #First Line exception for left grid line
                                    cv2.line(image,(nextColumnLeft, columnSentences[0]['top']), (nextColumnLeft, nextLineTop),(0, 255, 0), 2)
                                else: #Left and right lines
                                    cv2.line(image,(currentColumnLeft, columnSentences[0]['top']), (currentColumnLeft, nextLineTop),(0, 255, 0), 2)
                                    #Possible Right side Column Line
                                    if ((sentIndex + 1 ) < len(columnSentences)):
                                        if(columnLeftGap['cLineNo'] == columnSentences[sentIndex + 1]['cLineNo']):
                                            cv2.line(image,(nextColumnLeft, columnSentences[0]['top']), (nextColumnLeft, nextLineTop),(0, 255, 0), 2)
                    
                    cv2.line(image, (Lx,columnSentences[0]['top']), (Bx,columnSentences[0]['top']), (0,255,0), 2)
        print("---Execution Time for Applying Boundary Lines %s seconds ---" % (time.time() - start_time))
        # write_path = "/Users/swathi/documents/UpdatedOutput/" + " proccessed " + image_name
        # cv2.imwrite(write_path, image)
        return image

    def checkAndUpdateCurrentShape(self,currentShape, newShape):
        if (currentShape != newShape):
            return newShape
        else:
            return ""

    def getShape(self, text):
        numRegex = '[0-9]'
        lowercaseRegex = '[a-z]'
        uppercaseRegex = '[A-Z]'
        currencyRegex = '[$£]'

        shape = ""
        currentShape = ""
        for i in range(0, len(text)):
            currentChar = text[i]
            if re.match(numRegex, currentChar):
                shape = shape + \
                    self.checkAndUpdateCurrentShape(currentShape, "d")
                currentShape = "d"
            elif re.match(lowercaseRegex, currentChar):
                shape = shape + \
                    self.checkAndUpdateCurrentShape(currentShape, "x")
                currentShape = "x"
            elif re.match(uppercaseRegex, currentChar):
                shape = shape + \
                    self.checkAndUpdateCurrentShape(currentShape, "X")
                currentShape = "X"
            elif re.match(currencyRegex, currentChar):
                shape = shape + \
                    self.checkAndUpdateCurrentShape(currentShape, "c")
                currentShape = "c"
            else:
                shape = shape + currentChar
                currentShape = currentChar
        shape = shape.replace("d,d", "t")
        if shape == "d" and len(text) == 4:
            if re.match('^(?:19|20)\d{2}$', text):
                shape = "y"
        # Hanlding only text
        if ("x" in shape.lower()):
            if (not re.match(numRegex, shape)) and (not re.match(currencyRegex, shape)):
                shape = "x"
        return shape

#Modification ends
"""