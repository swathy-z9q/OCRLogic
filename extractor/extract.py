import numpy as np

import pandas as pd
from pandas import DataFrame

import pytesseract
from pytesseract import Output
import cv2
import os
import re
import json
import uuid
import time


from adapters.s3_adapter import S3_Adapter
from config.config import *
from imageprocessing.preprocessing import preprocessing
from imageprocessing.postProcessing import imagePostProcessing
from utils.ocrUtils import ocrUtil
from utils.pdfHelper import pdfConverter
from log.logWrapper import customlogger, logging

clog = customlogger(logging.getLogger(__name__)).getLogger()
regex = re.compile('[-.@_!#$%^&*()<>?/\|}{~:]')  

class UnstructuredExtractor:
    def __init__(self, debug=False):
        self.debug = debug
    def getImageRotationData(self):
        directory = "/Users/swathi/Documents/OrientationCheck"
        image_name = []
        rot_datas = []
        for filename in os.listdir("/Users/swathi/Documents/OrientationCheck"):
            if filename.endswith(".jpg"): 
                page_path = os.path.join(directory, filename)
                print('File Processed',filename)
                image = cv2.imread(page_path)
                image_name.append(filename)
                rot_datas.append(pytesseract.image_to_osd(image))
        rotationDf = { 'Image': pd.Series(image_name), 'RotationInfo': pd.Series(rot_datas) } 
        result = pd.DataFrame(rotationDf) 
        return result

    #def preprocessImage(self, doc, debug, debug_page_path):
    #    imgPreProcessing = preprocessing()
    #    doc = imgPreProcessing.removeHorizontalLines(doc, debug, debug_page_path.split('.jpg')[0] + "_hlremoved.jpg")
    #    doc = imgPreProcessing.removeVerticalLines(doc, debug, debug_page_path.split('.jpg')[0] + "_vlremoved.jpg")
    #    doc = imgPreProcessing.increaseContrast(doc, debug, debug_page_path.split('.jpg')[0] + "_contrast.jpg")
    #    d = pytesseract.image_to_data(doc, lang='eng', config="--oem 3 --psm 6", output_type=Output.DICT)
    #   return d,doc #OCR words, image
    
    
    def preProcessImage(self, doc, debug, debug_page_path, isDeskewed = False):
        
        imgPreProcessing = preprocessing()
        doc = imgPreProcessing.removeHorizontalLines(doc, debug, debug_page_path.split('.jpg')[0] + "_hlremoved.jpg")
        doc = imgPreProcessing.removeVerticalLines(doc, debug, debug_page_path.split('.jpg')[0] + "_vlremoved.jpg")
        doc = imgPreProcessing.increaseContrast(doc, debug, debug_page_path.split('.jpg')[0] + "_contrast.jpg")
        start_time = time.time()
        d = pytesseract.image_to_data(doc, lang='eng', config="--oem 3 --psm 6", output_type=Output.DICT)
        end_time = time.time()
        print('For Image to data', end_time-start_time)
        #isOcrHighConfidence = True if np.median([ int(d['conf'][i]) for i in range(len(d['level']))]) > 60 else False
        #if(isOcrHighConfidence or isDeskewed):
        return doc,d
        #if(not isOcrHighConfidence and not isDeskewed):
        #    doc = imgPreProcessing.rotate_bound(doc,180)
        #    return self.preProcessImage(doc,debug,debug_page_path,True)

        #return doc,d
        

    def run_extraction_process(self, page_path, image_name):
        if os.path.exists(page_path):
            # load the input image and grab the image dimensions
            doc = cv2.imread(page_path, 0)
            #image_name = page_path.partition('data/')[2].rsplit('/',1)[0]
            debug_page_path = '/Users/swathi/Documents/debug/'+image_name
            #contrast_page_path = '/Users/swathi/Documents/contrasted_images/' + image_name
            
            #rotationDf = self.getImageRotationData()
            #rotationDf.to_csv('/Users/swathi/Documents/rotationResult.csv', index=False)
            #doc = imgPreProcessing.removeHorizontalLines(doc, self.debug, debug_page_path.split('.jpg')[0] + "_hlremoved.jpg")
            #doc = imgPreProcessing.removeVerticalLines(doc, self.debug, debug_page_path.split('.jpg')[0] + "_vlremoved.jpg")
            #doc = imgPreProcessing.increaseContrast(doc, self.debug, contrast_page_path.split('.jpg')[0] + "_contrast.jpg")
            #contrastedImage = doc.copy()
            #d = pytesseract.image_to_data(doc, lang='eng', config="--oem 3 --psm 6", output_type=Output.DICT)
            #doc = imgPreProcessing.removeBg(doc,image_name)

            imgPreProcessing = preprocessing()
            #start_time = time.time()
            #doc = imgPreProcessing.orientImage(doc)
            #end_time = time.time()
            #print('For orientation Info', end_time-start_time)
            doc,d = self.preProcessImage(doc,self.debug, debug_page_path.split('.jpg')[0] + "_processeed.jpg")
            s = ocrUtil().getSentencesFromData(d)
            l = self.manipulateAndFindLines(s,doc.shape)
            #l, tableId = self.findStructureOfTable(l, doc)
            for lineNo,line in enumerate(l):
                for sentNo,sent in enumerate(line['sentences']):
                    #for wordNo,word in enumerate(sent['words']):
                    if(sent['isPageNo'] == True):
                        (x, y, w, h) = (sent['left'], sent['top'], sent['width'], sent['height'])
                        cv2.putText(doc,str('Page No'), (x - 50, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)

           

            if self.debug:  
                imgPostProcessing = imagePostProcessing()
                
                #mHeight,mCharWidth   = imgPostProcessing.getHeaders(d)
                #thresh = imgPostProcessing.displayOCRWords(d, doc, image_name)
                doc = imgPostProcessing.getBoldText(l, doc, image_name)

                #nonEmptyI = [i for i in range(len(d['text'])) if len(d['text'][i]) > 0 and d['conf'][i] >= 60]
                
                write_path = "/Users/swathi/documents/fin/" + image_name + ".jpg"
                cv2.imwrite(write_path, doc)
    
                #doc = imgPostProcessing.displayImageSentences(s, doc)
                #doc = imgPostProcessing.drawTable(l, doc, False) #image_name --> False
                
                #doc1 = np.zeros(doc.shape,dtype=np.uint8)
                #doc1.fill(255)
                #doc1 = imgPostProcessing.drawImageSentences(s, doc1)
                #doc1 = imgPostProcessing.drawTable(l, doc1, False)
                #cv2.imwrite(debug_page_path.split('.jpg')[0] + "_created.jpg", doc1)
               # cv2.imwrite(page_path.split('.jpg')[0] + "_modified.jpg", doc)

            #json_file_name = page_path.split('.jpg')[0] + ".json"
            #s3_json_path = S3_DIR + folder_name + '/' + json_file_name.split('/')[-1]
            #result = self.generateOutputJSON(l)
            #f = open(json_file_name, "w")
            #f.write(result)
            #f.close()
            # S3_Adapter().upload(json_file_name, s3_json_path)
            # if os.path.exists(page_path):
            #     os.unlink(page_path)
            
            #if os.path.exists(json_file_name):
            #    os.unlink(json_file_name)
            return True#s3_json_path
        else:
            raise Exception("Image file does not exist in path %s" %s (page_path))

    def isBetween(self, y1a, y2a, y1b, y2b):
        if(y1a == y1b and y2a == y2b):
            return True
        return ((y1a < y1b < y2a) or (y1a < y2b < y2a)) or (y1b < y1a < y2b) or (y1b < y2a < y2b)
    
   
    #In whole Page top: within 15% of the total page height
    #           bottom: Sentence present outside the 90% of the height
    def isPageNumber(self,sentence,sentNo,LineSentences,image_dimension):
        
        imgHeight,imgWidth = image_dimension[0],image_dimension[1]

        if(sentNo>0):
            prevSentence = LineSentences[sentNo-1]
            if(prevSentence['cLineNo'] != sentence['cLineNo']):
                return True
            
            
            #if(prevSentence['cLineNo'] ==  sentence['cLineNo'] and abs(prevSentence['right'] - sentence['left']) >= 2000):
            #    return True
            
        if(sentNo+1 < len(LineSentences) ):
            nextSentence = LineSentences[sentNo+1]
            #if(nextSentence['cLineNo'] ==  sentence['cLineNo'] and abs(nextSentence['left'] - sentence['right']) >= 2000):
            #    return True
            if(nextSentence['cLineNo'] != sentence['cLineNo']):
                return True
            else:
                pass
        return False    

    def manipulateAndFindLines(self, sentences, image_dimension):
        lineNo = 1
        availableLines = {}
        lines = []
        sentences = sorted(sentences, key=lambda i: i['top'])
        for sentNo,sentence in enumerate(sentences):
            if(sentence['cLineNo'] == None and sentence['text'] != ''):
                for otherSentence in sentences:
                    if((otherSentence['cLineNo'] == None and otherSentence['text'] != '') and (sentence['left'] != otherSentence['left'])):
                        if(self.isBetween(sentence['top'], sentence['top'] + sentence['height'], otherSentence['top'], otherSentence['top'] + otherSentence['height'])):
                            sentence['cLineNo'] = lineNo
                            sentence['potentialTable'] = True
                            otherSentence['cLineNo'] = lineNo
                            otherSentence['potentialTable'] = True
                sentence['cLineNo'] = lineNo
                lineNo += 1
            if sentence['cLineNo'] in availableLines:
                availableLines[sentence['cLineNo']].append(sentence)
            else:
                availableLines[sentence['cLineNo']] = [sentence]
           

            #            if re.match("(^(page)\s{0,3}[\d]+$)", sentence['text'], re.IGNORECASE):
            #                sentence['NumberNo'] = True

        # Logic to identify the page no
        # Case 1 : If any sentence has "Page" followed by number with other symbols like ":- " in between then it is potential page no
        # Case 2 : If there is a right aligned sentence with a number in the first two rows and no other sentence is above the sentence
        # Case 3 : Last line of the page has only sentence or right aligned sentence with only number

        # Reforming the line as array of dict
        for lineIdx in availableLines:
            lineSentences = sorted(availableLines[lineIdx], key=lambda i: i['left'])
            firstSentence = lineSentences[0]
            lastSentence = lineSentences[len(lineSentences) - 1]
            lineLeft = firstSentence['left']
            lineTop = ocrUtil.getTopFromWords(lineSentences)
            lineWidth = (lastSentence['left'] +
                         lastSentence['width']) - lineLeft
            lineHeight = ocrUtil.getHeightFromWords(lineSentences)
            combinedText = ' '.join([item['text'] for item in lineSentences])
            matched = re.match(
                "((^[\d]+|^[(]?[a-zA-Z]|(^[(]?[iI]+))\s*[.|)])(.*?)(?=([\d]+\.)|($))", combinedText)
            isLineTable = False
            isLineList = False
            for sentence in lineSentences:
                if matched:
                    sentence['bulleted'] = True
                    sentence['potentialTable'] = False
                    isLineList = True
                elif sentence['isPageNo'] == False and sentence['potentialTable'] == True:
                    isLineTable = True
                    break
            lines.append({'lineNo': lineIdx, 'sentences': lineSentences, 'left': lineLeft, 'top': lineTop, 'height': lineHeight,
                          'width': lineWidth, 'potentialTable': isLineTable, 'potentialList': isLineList, 'tableId': -1})

        if len(lines) > 0:
            pageLeft = ocrUtil.getMedianFromArrayOfDict(lines, 'left')
            pageWidth = ocrUtil.getMedianFromArrayOfDict(lines, 'width', False, False)
            pageRight = pageLeft + pageWidth

        # If there is only one sentence in a line and if it is right aligned, it is a potential table
        for lineIdx in range(len(lines)):
            currentLineIsList = lines[lineIdx]['potentialList']
            if not currentLineIsList:
                currentLineIsTable = lines[lineIdx]['potentialTable']
                if not currentLineIsTable:
                    noOfSentencesInLine = len(lines[lineIdx]['sentences'])
                    if noOfSentencesInLine == 1:
                        # Check if the sentence is right aligned
                        sentence = lines[lineIdx]['sentences'][0]
                        # if abs((sentence['width'] + sentence['left']) - pageRight) <= 300:
                        if sentence['left'] >= (pageLeft + (pageWidth / 2)) and len(sentence['text']) > 3:
                            sentence['potentialTable'] = True
                            lines[lineIdx]['potentialTable'] = True

        # If there is a potential table line between 2 non tables, then the current line will also be non table
        for lineIdx in range(len(lines)):
            currentLineIsList = lines[lineIdx]['potentialList']
            if not currentLineIsList:
                currentLineIsTable = lines[lineIdx]['potentialTable']

                prevLineIsTable = False
                nextLineIsTable = False
                if lineIdx != 0:
                    prevLineIsTable = lines[lineIdx - 1]['potentialTable']
                if lineIdx != len(lines) - 1:
                    nextLineIsTable = lines[lineIdx + 1]['potentialTable']

                # Handling the header scenario
                if (lineIdx == 0 and currentLineIsTable and nextLineIsTable == False):
                    lines[lineIdx]['potentialTable'] = False
                    for sentence in lines[lineIdx]['sentences']:
                        sentence['potentialTable'] = False

                if (currentLineIsTable == False and (prevLineIsTable == True and nextLineIsTable == True)):
                    if (lines[lineIdx]['width'] < pageWidth * (2/3)):
                        # Need to check if any of the sentence range (width) matches
                        # Need to check if height matches with prev and next
                        lines[lineIdx]['potentialTable'] = True
                        for sentence in lines[lineIdx]['sentences']:
                            sentence['potentialTable'] = True

        # for lineIdx in range(len(lines)):
        #     currentLineIsTable = lines[lineIdx]['potentialTable']
        #     currentLineIsNotTable = False
        #     if currentLineIsTable:
        #         for sentence in lines[lineIdx]['sentences']:
        #             sentence['right'] < pageLeft
        #             currentLineIsNotTable = True
        #             break
        #         if currentLineIsNotTable:
        #             lines[lineIdx]['potentialTable'] = False
        #             for sentence in lines[lineIdx]['sentences']:
        #                 sentence['potentialTable'] = False

        # If there are non potential lines between 2 tables, current lines can be considered as table
        # The lines should not be potential list
        # Lines should be less than half of the table length
        # no of non potential lines should not be greater than 2
        # Take distance between the lines into consideration
        for lineIdx in range(len(lines)):
            currentLineIsTable = lines[lineIdx]['potentialTable']
            if currentLineIsTable:
                nextLineIsTable = False
                nextLineIsList = False
                secondNextLineIsTable = False
                secondNextLineIsList = False
                thirdNextLineIsTable = False
                thirdNextLineIsList = False
                fourthNextLineIsTable = False
                fourthNextLineIsList = False

                try:
                    if lineIdx < len(lines) - 1:
                        nextLineIsTable = lines[lineIdx + 1]['potentialTable']
                        nextLineIsList = lines[lineIdx + 1]['potentialList']
                    if lineIdx < len(lines) - 2:
                        secondNextLineIsTable = lines[lineIdx +
                                                      2]['potentialTable']
                        secondNextLineIsList = lines[lineIdx +
                                                     2]['potentialList']
                    if lineIdx < len(lines) - 3:
                        thirdNextLineIsTable = lines[lineIdx +
                                                     3]['potentialTable']
                        thirdNextLineIsList = lines[lineIdx +
                                                    3]['potentialList']
                    if lineIdx < len(lines) - 4:
                        fourthNextLineIsTable = lines[lineIdx +
                                                      4]['potentialTable']
                        fourthNextLineIsList = lines[lineIdx +
                                                     4]['potentialList']

                    if ((not nextLineIsTable) and (not nextLineIsList)):
                        if (secondNextLineIsTable):
                            if (lines[lineIdx + 1]['width'] < pageWidth * (2/3)):
                                lines[lineIdx + 1]['potentialTable'] = True
                                for sentence in lines[lineIdx + 1]['sentences']:
                                    sentence['potentialTable'] = True
                        elif (not secondNextLineIsList):
                            if thirdNextLineIsTable:
                                if (lines[lineIdx + 1]['width'] < pageWidth * (2/3)) and (lines[lineIdx + 2]['width'] < pageWidth * (2/3)):
                                    lines[lineIdx + 1]['potentialTable'] = True
                                    lines[lineIdx + 2]['potentialTable'] = True
                                    for sentence in lines[lineIdx + 1]['sentences']:
                                        sentence['potentialTable'] = True
                                    for sentence in lines[lineIdx + 2]['sentences']:
                                        sentence['potentialTable'] = True
                            elif (not thirdNextLineIsList):
                                if fourthNextLineIsTable:
                                    if (lines[lineIdx + 1]['width'] < pageWidth * (2/3)) and (lines[lineIdx + 2]['width'] < pageWidth * (2/3)) and (lines[lineIdx + 3]['width'] < pageWidth * (2/3)):
                                        lines[lineIdx +
                                              1]['potentialTable'] = True
                                        lines[lineIdx +
                                              2]['potentialTable'] = True
                                        lines[lineIdx +
                                              3]['potentialTable'] = True
                                        for sentence in lines[lineIdx + 1]['sentences']:
                                            sentence['potentialTable'] = True
                                        for sentence in lines[lineIdx + 2]['sentences']:
                                            sentence['potentialTable'] = True
                                        for sentence in lines[lineIdx + 3]['sentences']:
                                            sentence['potentialTable'] = True
                except:
                    print('Exception occurred when identifying potential table lines')
        #To Identify Page Number 
        # 1. Identify the potential sentence that can be the page number in standard page number format
        # 2. Check for the sentence location towards the page position
        # 3. Check for the sentence for the location towards the page foramts
        try:
            lines = self.getPageNumber(lines,image_dimension)
        except:
            print('Exception occurred in identifying potential Page Number')

        return lines


    def getPageNumber(self,lines,image_dimension):
        for lineNo,line in enumerate(lines):
            lineSentences = line['sentences']
            for sentNo,sentence in enumerate(lineSentences):
                
                isPageNoFormat = False
                sentence['isPageNo'] =False
                imgHeight,imgWidth = image_dimension[0],image_dimension[1]

                #Checking sentence for page number formats like 1.Page 30 2.Page 10 of 20 3. 30
                if re.match("(^(page)\s{0,3}[\d]+$)", sentence['text'], re.IGNORECASE) or re.match("(^(page) ([0-9]+) ((of)|(-)) ([0-9]+))", sentence['text'], re.IGNORECASE) or (self.getShape(sentence['text'])=='d') :
                    isPageNoFormat = True
                    
                if(isPageNoFormat):
                    topLeftCorner = True if int(0.15 * imgWidth) > sentence['right'] and int(0.15 * imgHeight) > sentence['top'] else False
                    topRightCorner = True if int(0.85 * imgWidth) < sentence['right'] and int(0.15 * imgHeight) > sentence['top'] else False
                    bottomLeftCorner = True if int(0.15 * imgWidth) > sentence['right'] and int(0.90 * imgHeight) < sentence['top'] else False
                    bottomRightCorner = True if int(0.85 * imgWidth) < sentence['right'] and int(0.90 * imgHeight) < sentence['top'] else False
                    
                    #Check for all 4 corners:
                    if(topLeftCorner or topRightCorner):
                        sentence['isPageNo'] = True
                        line['potentialTable'] = False
                        break

                        #if(self.isPageNumber(sentence,sentNo,lineSentences,image_dimension)):
                    if(bottomLeftCorner or bottomRightCorner) and lineNo == len(lines)-1:
                        sentence['isPageNo'] = True
                        line['potentialTable'] = False
                        break
                            
                
                    #Check for top and bottom of the page
                    if( (int(0.15 * imgHeight) > sentence['top'] or (int(0.90 * imgHeight) < sentence['top']) and len(lineSentences) == 1) ):    
                        #if(self.isPageNumber(sentence,sentNo,lineSentences,image_dimension)):
                        sentence['isPageNo'] = True
                        line['potentialTable'] = False
                        break
                
                    #Check for first or last line of the page for page number centre or right alligned
                    midPage = int(imgWidth/2)
                    if((lineNo == len(lines)-1 or lineNo == 0) and len(lineSentences) == 1 and sentence['left'] >= (midPage-700) and sentence['right'] <= (midPage + 700)):
                        sentence['isPageNo'] = True
                        line['potentialTable'] = False
                        break
        
        return lines

    def findStructureOfTable(self, lines, image):
        # Grouping table rows as tables
        tableList = []
        tableRowList = []
        tableStarted = False

        # Get left, width of the entire page based on the left and width of lines
        if len(lines) > 0:
            pageLeft = ocrUtil.getMedianFromArrayOfDict(lines, 'left')
            pageWidth = ocrUtil.getMedianFromArrayOfDict(lines, 'width', False, False)

        for lineIdx in range(0, len(lines)):
            currentLineIsTable = lines[lineIdx]['potentialTable']

            if(tableStarted == False and currentLineIsTable == True):
                tableStarted = True
                tableRowList.append(lines[lineIdx])
            elif(tableStarted and currentLineIsTable == False):
                tableStarted = False
                tableList.append(tableRowList)
                tableRowList = []
            elif currentLineIsTable == True:
                tableRowList.append(lines[lineIdx])

        if len(tableRowList) > 0:
            tableList.append(tableRowList)

        # Logic to split the tables
        # Condition 1 : If the gap between rows greater than 6 times the average gap between rows
        for table in tableList:
            if len(table) > 2:
                tableRowGap = []
                for rowIdx in range(0, len(table)):
                    # Check if there is next row in the table
                    if rowIdx < len(table) - 1:
                        nextRow = table[rowIdx + 1]
                        currRow = table[rowIdx]
                        tableRowGap.append(
                            nextRow['top'] - (currRow['top'] + currRow['height']))
                aray = np.asarray(tableRowGap)
                m = np.median(aray)
                index = (np.abs(aray - m)).argmin()
                avgTableRowGap = int(aray[index])

                for rowIdx in range(0, len(table)):
                    # Check if there is next row in the table
                    if rowIdx < len(table) - 1:
                        nextRow = table[rowIdx + 1]
                        currRow = table[rowIdx]
                        if (nextRow['top'] - (currRow['top'] + currRow['height'])) > (6 * avgTableRowGap):
                            tableList.append(table[rowIdx+1:])
                            del table[rowIdx+1:]

        # Updating the table id
        tableId = 1
        for table in tableList:
            for rowIdx in range(0, len(table)):
                table[rowIdx]['tableId'] = tableId
            tableId += 1

        for table in tableList:
            if len(table) == 0:
                continue

            # If the table has only one row, then it is not a table
            if len(table) == 1:
                for rowIdx in range(0, len(table)):
                    table[rowIdx]['potentialTable'] = False
                    table[rowIdx]['tableId'] = -1
                    for sentence in table[rowIdx]['sentences']:
                        sentence['potentialTable'] = False
                continue

            # Get maximum columns in the table
            noOfColumns = 0
            noOfRows = len(table)
            rowWithMaxCol = None
            top = None
            bottom = None
            currentTableId = None
            firstRowNo = None
            for rowIdx in range(0, len(table)):
                if rowIdx == 0:
                    top = ocrUtil.getMedianFromArrayOfDict(
                        table[rowIdx]['sentences'], 'top')
                    currentTableId = table[rowIdx]['tableId']
                    firstRowNo = table[rowIdx]['lineNo'] - 1
                if rowIdx == len(table) - 1:
                    bottom = ocrUtil.getMedianFromArrayOfDict(
                        table[rowIdx]['sentences'], 'bottom')
                currentRowColumns = len(table[rowIdx]['sentences'])
                if currentRowColumns > noOfColumns:
                    noOfColumns = currentRowColumns
                    rowWithMaxCol = table[rowIdx]

            # Get left most of the table combining all first sentences
            leftSentences = []
            for line in table:
                leftSentences.append(sorted(line['sentences'], key=lambda i: i['left'])[0])
            left = ocrUtil.getMedianFromArrayOfDict(leftSentences, 'left')

            # Find how many sentences are between table left and left of rowWithMaxCol and create a dummy sentence and add to the row with max column
            noOfSentencesBeforeIdentifiedRow = 0
            sortedRowWithMaxColSentences = sorted(rowWithMaxCol['sentences'], key=lambda i: i['left'])
            lineWithOtherSentences = None
            sentencesToBeCloned = []
            for line in table:
                tempSenCount = 0
                tempSenToBeCloned = []
                for sentence in line['sentences']:
                    if (sentence['left'] >= left and sentence['right'] <= sortedRowWithMaxColSentences[0]['left']) or (sentence['left'] >= sortedRowWithMaxColSentences[len(sortedRowWithMaxColSentences) - 1]['right']):
                        tempSenCount += 1
                        tempSenToBeCloned.append(sentence)
                if tempSenCount > noOfSentencesBeforeIdentifiedRow:
                    noOfSentencesBeforeIdentifiedRow = tempSenCount
                    lineWithOtherSentences = line
                    sentencesToBeCloned = tempSenToBeCloned

            # Clone and add sentences in
            for sentence in sentencesToBeCloned:
                dummyWord = ocrUtil.createWord(
                    '', sentence['top'], sentence['left'], sentence['width'], sentence['height'], 100, rowWithMaxCol['lineNo'])
                newSentence = ocrUtil.createSentence(
                    [dummyWord], sentence['top'], sentence['left'], sentence['width'], sentence['height'], '', rowWithMaxCol['lineNo'], True)
                rowWithMaxCol['sentences'].append(newSentence)

            # Pick the row which has maximum no of columns
            # Loop the sentences
            firstColLeft = None
            colIdx = 1
            for column in sorted(rowWithMaxCol['sentences'], key=lambda i: i['left']):
                # Get all the sentences which is in range of the current sentence and update the column index
                columnLeft = column['left']
                minLeft = columnLeft
                columnRight = column['right']
                maxRight = columnRight
                column['columnNo'].append(colIdx)
                for row in table:
                    if row['lineNo'] != rowWithMaxCol['lineNo']:
                        for otherSegments in row['sentences']:
                            osLeft = otherSegments['left']
                            osRight = otherSegments['right']

                            if ((osLeft <= columnLeft <= osRight) or (columnLeft <= osLeft <= columnRight)):
                                otherSegments['columnNo'].append(colIdx)
                                if osLeft < minLeft:
                                    minLeft = osLeft
                                if osRight > maxRight:
                                    maxRight = osRight
                colIdx += 1

            # If the table has only 2 columns and always the first column value is invalid character then it is not a table
            if colIdx == 3:
                currentTableIsList = False
                for line in table:
                    if currentTableIsList == False and len(line['sentences']) == 2:
                        firstSentence = line['sentences'][0]
                        secondSentence = line['sentences'][1]
                        currentTableIsList = (len(firstSentence['text']) == 1 and self.getShape(
                            secondSentence['text']) == 'x')
                    if currentTableIsList:
                        for sentence in line['sentences']:
                            sentence['bulleted'] = True
                            sentence['potentialTable'] = False
                            sentence['columnNo'] = []
                        line['potentialTable'] = False
                        line['potentialList'] = True
                if currentTableIsList:
                    continue

            # Handling empty column id - If a column id is empty then check if there is anyother sentence is in the same range and update its column id
            for line in table:
                for sentence in line['sentences']:
                    if len(sentence['columnNo']) == 0:
                        senLeft = sentence['left']
                        senRight = sentence['right']
                        for row in table:
                            if row['lineNo'] != line['lineNo']:
                                for otherSegments in row['sentences']:
                                    if len(otherSegments['columnNo']) == 1:
                                        osLeft = otherSegments['left']
                                        osRight = otherSegments['right']

                                        if ((osLeft <= senLeft <= osRight) or (senLeft <= osLeft <= senRight)):
                                            if otherSegments['columnNo'][0] not in sentence['columnNo']:
                                                sentence['columnNo'].append(
                                                    otherSegments['columnNo'][0])
                                                break

            # Correct colspan
            # Calculate the x and y points
            # Construct a grid using the xpoints and ypoints which will be used in colspan fixing

            columns = {}
            padding = 5

            for line in table:
                # Loop all the sentences and add it to columns
                for sentence in line['sentences']:
                    if len(sentence['columnNo']) > 0:
                        colNo = sentence['columnNo'][0]
                        if colNo in columns.keys():
                            columns[colNo].append(sentence)
                        else:
                            columns[colNo] = [sentence]

            leftX = 0
            rightX = 0
            topY = 0
            bottomY = 0
            firstRow = True
            lastLine = None
            yPoints = {}

            yIndex = 1
            for line in sorted(table, key=lambda i: i['lineNo']):
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
            # currentTableColumns = [v for v in columns.items()][0]

            leftX = leftX - padding
            xPoints = {0: leftX}

            xIndex = 1
            for key in sorted(columns.keys(), key=lambda i: i):
                cols = columns[key]
                aray = np.asarray([col['right']
                                   for col in cols if len(col['columnNo']) == 1])
                if len(aray) == 0:
                    aray = np.asarray(
                        [col['right'] for col in cols if (xIndex in col['columnNo'])])
                xPoints[xIndex] = int(np.max(aray)) + padding
                xIndex += 1

            # Case 1 : If there is a sentence with potential table 'True' but no column identified it is potential colspan
            # Case 2 : If there is a sentence which has colspan but column boundaries are disecting the sentence then it is potential colspan
            for line in table:
                for sentence in line['sentences']:
                    sLeft = sentence['left']
                    sRight = sentence['right']
                    if len(sentence['columnNo']) == 0:
                        for xPointKey in xPoints:
                            if xPointKey != 0:
                                if xPoints[xPointKey - 1] <= sLeft and sRight <= xPoints[xPointKey]:
                                    sentence['columnNo'].append(xPointKey)
                    # Case 2
                    for xPointKey in xPoints:
                        if xPointKey != 0:
                            if sLeft <= xPoints[xPointKey] <= sRight:
                                if xPointKey not in sentence['columnNo']:
                                    sentence['columnNo'].append(xPointKey)
                            if xPoints[xPointKey - 1] <= sRight <= xPoints[xPointKey]:
                                if xPointKey not in sentence['columnNo']:
                                    sentence['columnNo'].append(xPointKey)
        return lines, tableId

    def generateOutputJSON(self, lines, formatted=False):
        if formatted:
            result = {
                'table': {},
                'list': [],
                'lines': [],
                'segments': [],
                'words': []
            }

            for line in lines:
                currentLineNo = line['lineNo']
                tableId = 'tableId_' + str(line['tableId'])
                if line['potentialTable'] == True:
                    # Find if column key is present
                    for segment in line['sentences']:
                        columnNo = 'columnId_' + str(segment['columnNo'])
                        currentSegment = {'id': segment['id'], 'top': segment['top'],
                                          'left': segment['left'], 'width': segment['width'], 'height': segment['height'], 'lineNo': currentLineNo, 'words': []}
                        for word in segment['words']:
                            currentWord = {'text': word['text'], 'top': word['top'], 'left': word['left'],
                                           'width': word['width'], 'height': word['height'], 'conf': word['conf'], 'id': word['id']}
                            result['words'].append(currentWord)
                            currentSegment['words'].append({'id': word['id']})
                        if tableId in result['table']:
                            if columnNo in result['table'][tableId]:
                                result['table'][tableId][columnNo]['segments'].append(
                                    {'id': segment['id']})
                            else:
                                result['table'][tableId][columnNo]['segments'] = [
                                    {'id': segment['id']}]
                        else:
                            result['table'][tableId] = {columnNo: {
                                'segments': [{'id': segment['id']}]}}
                        result['segments'].append(currentSegment)
                elif line['potentialList'] == True:
                    currentList = {'lineNo': currentLineNo, 'segments': []}
                    for segment in line['sentences']:
                        currentSegment = {'id': segment['id'], 'top': segment['top'],
                                          'left': segment['left'], 'width': segment['width'], 'height': segment['height'], 'lineNo': currentLineNo, 'words': []}
                        for word in segment['words']:
                            currentWord = {'text': word['text'], 'top': word['top'], 'left': word['left'],
                                           'width': word['width'], 'height': word['height'], 'conf': word['conf'], 'id': word['id']}
                            result['words'].append(currentWord)
                            currentSegment['words'].append({'id': word['id']})
                        currentList['segments'].append({'id': segment['id']})
                        result['segments'].append(currentSegment)
                else:
                    currentLine = {'lineNo': currentLineNo, 'left':  line['left'], 'top':  line['top'],
                                   'height':  line['height'], 'width':  line['width'], 'segments': []}
                    for segment in line['sentences']:
                        currentSegment = {'id': segment['id'], 'top': segment['top'],
                                          'left': segment['left'], 'width': segment['width'], 'height': segment['height'], 'lineNo': currentLineNo, 'words': []}
                        for word in segment['words']:
                            currentWord = {'text': word['text'], 'top': word['top'], 'left': word['left'],
                                           'width': word['width'], 'height': word['height'], 'conf': word['conf'], 'id': word['id']}
                            result['words'].append(currentWord)
                            currentSegment['words'].append({'id': word['id']})
                        currentLine['segments'].append({'id': segment['id']})
                        result['segments'].append(currentSegment)
                    result['lines'].append(currentLine)
            return json.dumps(result)
        else:
            return json.dumps(lines)

    
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

    def checkAndUpdateCurrentShape(self, currentShape, newShape):
        if (currentShape != newShape):
            return newShape
        else:
            return ""
