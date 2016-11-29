import re
import string
import random
import numpy as np
import json
import classes
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import pandas as pd
from baseModel import classifiers, classifier_names
from datetime import datetime

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

def loadData():
    data = classes.dataDict()
    dataDict = {}
    dataDict['DESCR'] = None 
    dataDict['data'] = data.contents
    dataDict['target_names'] = ['RED', 'YELLOW', 'GREEN']
    dataDict['target'] = data.grades
    dataDict['description'] = 'dataset of graded solicitations from 2009-2012'
    return dataDict

def kfolds_split(inputDict, numFolds):
    dataSetLength = len(dataDict['data'])
    indexArray = list(range(0, dataSetLength))
    random.shuffle(indexArray)
    return np.array_split(indexArray, numFolds)

def test_model_accuracy(splits, dataSet):
    results = {}
    for i in range(0, len(splits)):
        test_index = splits[i]
        print('processing fold ' + str(i))
        remaining = list(range(0, len(splits)))
        remaining.remove(i)
        train_index = []
        for x in remaining:
            train_index.append(splits[x])
        train_index = np.concatenate(train_index)
        data_train = dataHandling.makeSubDict(dataSet, test_index)
        data_test = dataHandling.makeSubDict(dataSet, train_index)
        y_train, y_test = data_train['target'], data_test['target']
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        X_train = vectorizer.fit_transform(data_train['data'])
        X_test = vectorizer.transform(data_test['data'])
        resultDict = {}
        for j, clf in enumerate(classifiers):
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            score = metrics.accuracy_score(y_test, pred)
            resultDict[classifier_names[j]] = score
        results[i] = resultDict
    testOutput = pd.DataFrame.from_dict(results)
    testOutput['avg'] = testOutput.mean(axis=1)
    testOutput.to_csv('kfolds_model_accuracy_' + datetime.today().strftime("%Y%m%d") + '.csv')
    t = testOutput[['avg']]
    return t.to_dict()