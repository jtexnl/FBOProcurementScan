import re
import string
import random
import numpy as np
import json

def remove_punctuation(text):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    s = regex.sub(' ', text)
    return s

def remove_formatting(text):
    output = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').replace('\x0b', ' ').replace('\x0c', ' ')
    return output

def transform_for_classifier(text):
    temp = remove_punctuation(text)
    final = remove_formatting(temp)
    return final

def create_list_from_index(inputList, indexArray):
    outputList = []
    for i in indexArray:
        outputList.append(inputList[i])
    return outputList

def make_train_test_index(dataDict, testPct):
    dataSetLength = len(dataDict['data'])
    indexArray = list(range(0, dataSetLength))
    random.shuffle(indexArray)
    trainIndex = indexArray[0:int((len(indexArray) - (len(indexArray) * testPct)))]
    testIndex = indexArray[len(trainIndex):len(indexArray)]
    return trainIndex, testIndex

def make_sub_dict(mainDict, index):
    outputDict = {}
    for key in mainDict.keys():
        if type(mainDict[key]) is list and len(mainDict[key]) > 3:
            outputDict[key] = create_list_from_index(mainDict[key], index)
        elif type(mainDict[key]) is np.ndarray:
            outputDict[key] = create_list_from_index(mainDict[key], index)
        else:
            outputDict[key] = mainDict[key]
    return outputDict

def make_train_test(inputDict, testPct):
    trainIndex, testIndex = make_train_test_index(inputDict, testPct)
    testDict = make_sub_dict(inputDict, testIndex)
    trainDict = make_sub_dict(inputDict, trainIndex)
    return trainDict, testDict

def kfolds_split(dataDict, numFolds):
    dataSetLength = len(dataDict['data'])
    indexArray = list(range(0, dataSetLength))
    random.shuffle(indexArray)
    return np.array_split(indexArray, numFolds)

def writeJson(inputData, fileName):
    with open(fileName, 'w+') as outfile:
        json.dump(inputData, outfile, indent = 4)