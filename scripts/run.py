import classes
import os
from spacy.en import English
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import numpy as np
import json

#Initialize the dailySolicitationListing class to get the list of new solicitations
print('Loading raw data')
rawData = classes.dailySolicitationListing()

#Go through rawData and instatiate a newSolicitation() object for each line
print('scraping and parsing solicitation documents')
solicitations = []
for i in range(0, len(rawData.raw)):
    solicitations.append(classes.newSolicitation(rawData.raw, i))

### This section may not be necessary with the addition of predictionGenerator() object to classes. Check in.
print('loading predictors')
ridge = joblib.load('binaries/ridge.pkl')
nearestCentroid = joblib.load('binaries/nearestCentroid.pkl')
L2SVC = joblib.load('binaries/L2SVC.pkl')
BNB = joblib.load('binaries/BNB.pkl')
knn = joblib.load('binaries/knn.pkl')
perceptron = joblib.load('binaries/perceptron.pkl')
pipeline = joblib.load('binaries/pipeline.pkl')
L2SGD = joblib.load('binaries/L2SGD.pkl')
elasticNet = joblib.load('binaries/elasticNet.pkl')
L1SVC = joblib.load('binaries/L1SVC.pkl')
MNB = joblib.load('binaries/MNB.pkl')
L1SGD = joblib.load('binaries/L1SGD.pkl')
passiveAggressive = joblib.load('binaries/passiveAggressive.pkl')
randomForest = joblib.load('binaries/randomForest.pkl')

predictors = {'ridge':ridge, 'nearestCentroid':nearestCentroid, 'L2SVC':L2SVC, 'BNB':BNB, 'knn':knn, 'perceptron':perceptron, 'pipeline':pipeline, 'L2SGD':L2SGD, 'elasticNet':elasticNet, 'L1SVC':L1SVC, 'MNB':MNB, 'L1SGD':L1SGD, 'passiveAggressive':passiveAggressive, 'randomForest':randomForest}

predictionList = []
for i in range(0, len(solicitations)):
    subDict = {}
    subDict['url'] = solicitations[i].url
    subDict['text'] = solicitations[i].tokenized
    subDict['predictions'] = {}
    predictionList.append(subDict)

predictionText = []
for item in predictionList:
    predictionText.append(item['text'])

print('vectorizing text')
vec = joblib.load('binaries/vectorizer.pkl')

predictionVector = vec.transform(predictionText)

print('generating predictions')
#generate predictions
for name, predictor in predictors.items():
    results = predictor.predict(predictionVector)
    for i in range(0, len(results)):
        predictionList[i]['predictions'][name] = results[i]

for item in predictionList:
    values = list(item['predictions'].values())
    item['avgPred'] = np.mean(values)

#reformat the predictions for the formatted output
raw = {}
for i in predictionList:
   rel = i['predictions']
   for key, value in rel.items():
    if not key in raw:
        raw[key] = [value]
    else:
        raw[key].append(value)

formatted = classes.formattedPredictionOutput(raw, solicitations)

dataHandling.writeJson(formatted.finalOutput, 'predictions/predictions_' + str(rawData.date) + '.json')
