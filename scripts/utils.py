from sklearn.externals import joblib

class predictionGenerator():

    def __init__(self, inputData):
        self.algorithms = {'ridge' : joblib.load('ridge.pkl'),'nearestCentroid' : joblib.load('nearestCentroid.pkl'), 
        'L2SVC' : joblib.load('L2SVC.pkl'), 'BNB' : joblib.load('BNB.pkl'), 'knn' : joblib.load('knn.pkl'),
        'perceptron' : joblib.load('perceptron.pkl'), 'pipeline' : joblib.load('pipeline.pkl'), 'L2SGD' : joblib.load('L2SGD.pkl'), 
        'elasticNet' : joblib.load('elasticNet.pkl'), 'L1SVC' : joblib.load('L1SVC.pkl'), 'MNB' : joblib.load('MNB.pkl'), 
        'L1SGD' : joblib.load('L1SGD.pkl'), 'passiveAggressive' : joblib.load('passiveAggressive.pkl'),
        'randomForest' : joblib.load('randomForest.pkl')}
        self.vectorizer = joblib.load('vectorizer.pkl')
        self.vectorizedData = self.vectorizeData(inputData)
        self.predictionSet = self.generatePredictions(self.algorithms, self.vectorizedData)
    
    def vectorizeData(self, inputData):
        return self.vectorizer.transform(inputData)
    
    def generatePredictions(self, algorithms, vectorizedData):
        predictionDict = {}
        for algorithmName, algorithm in self.algorithms.items():
            predictionDict[algorithmName] = algorithm.predict(vectorizedData)
        return predictionDict

class formattedPredictionOutput():

    def __init__(self, rawPredictions, solicitationList):
        self.rawPredictions = rawPredictions
        self.solicitationList = solicitationList
        self.accuracyDict = self.load_accuracy_dict()
        self.qualitativePredictions = self.convert_to_qualitative(self.rawPredictions)
        self.maxScore = self.calculate_max_score()
        self.gradesBySolicitation = self.break_out_grades()
        self.finalOutput = self.combine_information()

    def load_accuracy_dict(self):
        with open('modelAccuracy.json', 'rU') as infile:
            data = json.load(infile)
        return data

    def convert_to_qualitative(self, rawPredictions):
        newDict = {}
        for key in list(self.accuracyDict.keys()):
            subDict = {}
            subDict['accuracy'] = self.accuracyDict[key]
            valueList = list(rawPredictions[key])
            scores = []
            for item in valueList:
                if item == 0:
                    scores.append('RED')
                elif item == 1:
                    scores.append('YELLOW')
                else:
                    scores.append('GREEN')
            subDict['scores'] = scores
            newDict[key] = subDict
        return newDict

    def calculate_max_score(self):
        maxScore = 0
        for key, value in self.qualitativePredictions.items():
            maxScore += self.qualitativePredictions[key]['accuracy']
        return maxScore

    def break_out_grades(self):
        final = {}
        for i in range(0, len(self.solicitationList)):
            solList = []
            for key, value in self.qualitativePredictions.items():
                subList = [self.qualitativePredictions[key]['scores'][i], self.qualitativePredictions[key]['accuracy']]
                solList.append(subList)
            grades = {}
            for row in solList:
                if row[0] in grades.keys():
                    grades[row[0]] += row[1]
                else:
                    grades[row[0]] = row[1]
            possibleGrades = ['RED', 'GREEN', 'YELLOW']
            for grade in possibleGrades:
                if grade in grades.keys():
                    continue
                else:
                    grades[grade] = 0
            subDict = {}
            for key, value in grades.items():
                subDict[key] = grades[key] / self.maxScore
            final[i] = subDict
        return final

    def combine_information(self):
        finalOutput = []
        for i in range(0, len(self.solicitationList)):
            subDict = {}
            subDict['url'] = solicitations[i].url
            subDict['agency'] = solicitations[i].agency
            subDict['predictions'] = self.gradesBySolicitation[i]
            finalOutput.append(subDict)
        return finalOutput