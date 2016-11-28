import os
import numpy as np
import re
import string
import random
import dataHandling
import rejectList


class dataDict():
    
    def __init__ (self):
        self.directory = os.listdir()
        self.directoryFiles = self.findRelevantFiles(self.directory)
        self.grades = self.buildGradesArray(self.directoryFiles)
        self.paths = self.buildPathsArray(self.directoryFiles)
        self.contents = self.buildContentsList(self.directoryFiles)
        
    def findRelevantFiles(self, directory):
        relevant = []
        for fileName in self.directory:
            if fileName[-3:] == 'txt':
                relevant.append(fileName)
        return relevant
    
    def buildGradesArray(self, directoryFiles):
        grades = []
        for i in directoryFiles:
            grade = i.split('_')[0]
            if grade == 'GREEN':
                gradeNum = 2
            elif grade == 'YELLOW':
                gradeNum = 1
            else:
                gradeNum = 0
            grades.append(gradeNum)
        gradeArray = np.asarray(grades)
        return gradeArray
    
    def buildPathsArray(self, directoryFiles):
        paths = []
        for i in directoryFiles:
            path = os.getcwd() + '/' + i
            paths.append(path)
        pathArray = np.asarray(paths)
        return pathArray
    
    def buildContentsList(self, directoryFiles):
        contentsList = []
        counter = 0
        for fileName in os.listdir():
            if fileName[-3:] == 'txt':
                print('processing: ' + str(counter) + ' Name: ' + fileName )
                counter += 1
                raw = open(fileName, encoding = 'Latin-1').read()
                clean1 = dataHandling.transform_for_classifier(raw)
                clean = rejectList.cleanUpText(clean1, rejectList.rejectList)
                contentsList.append(clean)
        return contentsList

def loadData():
    data = dataDict()
    dataDict = {}
    dataDict['DESCR'] = None 
    dataDict['data'] = data.contents
    dataDict['target_names'] = ['RED', 'YELLOW', 'GREEN']
    dataDict['target'] = data.grades
    dataDict['description'] = 'dataset of graded solicitations from 2009-2012'
    return dataDict
