import numpy as np
import uuid

class ocrUtil:

    def getMedianFromArrayOfDict(arr, key, considerScore=False, min=True):
        if considerScore:
            aray = np.asarray([item[key] for item in arr if item['conf'] > 80])
            if len(aray) == 0:
                aray = np.asarray([item[key] for item in arr])
        else:
            aray = np.asarray([item[key] for item in arr])
        m = np.median(aray)
        if min:
            index = (np.abs(aray - m)).argmin() 
        else:
            index = (aray - m).argmax()
        return int(aray[index])

    def getHeightFromWords(arr):
        aray = np.asarray([item['height'] for item in arr if item['height'] > 15])
        if len(aray) == 0:
            return None
        m = np.median(aray)
        index = (np.abs(aray - m)).argmin() 
        return int(aray[index])

    def getWidthFromWords(arr):
        aray = np.asarray([item['width'] for item in arr])
        return int(np.max(aray))

    def getTopFromWords(arr):
        aray = np.asarray([item['top'] for item in arr if item['height'] > 15])
        if len(aray) == 0:
            return None
        m = np.median(aray)
        index = (np.abs(aray - m)).argmin() 
        return int(aray[index])

    def getTextFromArrayOfDict(arr):
        return (' '.join([item['text'].strip() for item in arr if item['conf'] > 5])).strip()
    
    def __getSentenceFromWords(self, words):
        if words is not None and len(words) > 0:
            firstWord = words[0]
            lastWord = words[len(words) - 1]
            sentenceLeft = firstWord['left']
            sentenceTop = ocrUtil.getTopFromWords(words)
            sentenceWidth = (lastWord['left'] + lastWord['width']) - sentenceLeft
            sentenceHeight = ocrUtil.getHeightFromWords(words)
            text = ocrUtil.getTextFromArrayOfDict(words)
            # If first word is invalid and second word is number or text choose the second word
            if sentenceHeight == None or sentenceHeight < 15:
                return None
            elif text == '':
                return None
            elif (len(text) == 1 and text in ['.', ',']):
                return None
            else:
                return ocrUtil.createSentence(words, sentenceTop, sentenceLeft, sentenceWidth, sentenceHeight, text, None, False, [], False, False)
        return None
    
    def createSentence(words, sentenceTop, sentenceLeft, sentenceWidth, sentenceHeight, text, cLineNo = None, potentialTable = False, columnNo = [], bulleted=False, isPageNo = False, id=str(uuid.uuid1())):
        return {'words': words, 'top': sentenceTop, 'text': text, 'left': sentenceLeft, 'width': sentenceWidth, 'right': sentenceLeft + sentenceWidth, 
                            'bottom': sentenceTop + sentenceHeight, 'height': sentenceHeight, 'cLineNo': cLineNo, 'potentialTable': potentialTable, 'isPageNo': isPageNo, 'columnNo': columnNo, 'bulleted': bulleted, 'id': id}
    
    def createWord(text, top, left, width, height, conf, lineNum, id=str(uuid.uuid1())):
        return {'text': text, 'top': top, 'left': left, 'width': width, 'height': height, 'conf': conf, 'lineNum': lineNum, 'id': id}

    def getSentencesFromData(self, data):
        lineStarted = False
        sentences = []
        wordsInSentence = []
        lineNum = 0
        previousWord = None

        # To avoid vertical line noices
        invalidWordIndex = set()
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            if(data['level'][i] == 5):
                if (len(data['text'][i]) <= 1):
                    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    left = x
                    top = y
                    width = w
                    # Get all words which has same left
                    for j in range(n_boxes):
                        if(data['level'][j] == 5):
                            if abs(data['left'][j] - left) < 5 and abs(data['top'][j] - top) > 10 and abs(data['width'][j] - w) < 5 and w < 10 and len(data['text'][j]) <= 1:
                                if len(invalidWordIndex) == 0:
                                    invalidWordIndex.add(i)
                                invalidWordIndex.add(j)

        for index, level in enumerate(data['level']):
            if(lineStarted and level == 4):
                sentence = self.__getSentenceFromWords(wordsInSentence)
                if sentence != None:
                    sentences.append(sentence)
                wordsInSentence = []
                lineNum += 1
                previousWord = None
            elif (level == 4):
                lineStarted = True
            elif (lineStarted and level == 5):
                if index not in invalidWordIndex:
                    if data['height'][index] > 100 and (len(data['text'][index]) < 2 and data['conf'][index] < 5):
                        continue
                    elif data['width'][index] > 500 and len(data['text'][index].strip()) < 5:
                        continue

                    currentWord = ocrUtil.createWord(text=data['text'][index], top=data['top'][index], left=data['left'][index], width=data['width'][index], height=data['height'][index], conf=data['conf'][index], lineNum=lineNum)
                    
                    if(previousWord == None):
                        previousWord = currentWord
                    else:
                        # If the distance between words in the line is > threshold, split the lines as sentences
                        if(currentWord['left']) - (previousWord['left'] + previousWord['width']) > 50:
                            sentence = self.__getSentenceFromWords(wordsInSentence)
                            if sentence != None:
                                sentences.append(sentence)
                            wordsInSentence = []
                            previousWord = None
                        previousWord = currentWord
                    wordsInSentence.append(currentWord)
                else:
                    print(data['text'][index])
            else:
                continue

        sentence = self.__getSentenceFromWords(wordsInSentence)
        if sentence != None:
            sentences.append(sentence)
        return sentences