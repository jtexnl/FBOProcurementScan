import classes
import os
from spacy.en import English
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import numpy as np
import json

#Run the bash script to pull the most recent FBO Data Dump
os.system('bash pull/fbo-nightly.sh')

#Initialize the dailySolicitationListing class to get the list of new solicitations
print('Loading raw data')
rawData = classes.dailySolicitationListing()

#Create a new directory where the documents will be stored until parsed. This directory will be deleted at the end
os.mkdir('temp_test')
#Move into the temporary directory
os.chdir('temp_test')
print('scraping and parsing solicitation documents')
solicitations = []
for i in range(0, len(rawData.raw)):
    print('scraping solicitation number ' + str(i) + ' for documents')
    solicitations.append(classes.solicitation_documents(rawData.raw[i]))

os.chdir('..')
os.system('rm -r temp_test')
latest_update = open('latest_update.txt').read()
### This section may not be necessary with the addition of predictionGenerator() object to classes. Check in.
print('loading predictors')
ridge = joblib.load('binaries/' + latest_update + '/ridge.pkl')
logit = joblib.load('binaries/' + latest_update + '/logit.pkl')
nearestCentroid = joblib.load('binaries/' + latest_update + '/nearestCentroid.pkl')
L2SVC = joblib.load('binaries/' + latest_update + '/L2SVC.pkl')
BNB = joblib.load('binaries/' + latest_update + '/BNB.pkl')
knn = joblib.load('binaries/' + latest_update + '/knn.pkl')
perceptron = joblib.load('binaries/' + latest_update + '/perceptron.pkl')
pipeline = joblib.load('binaries/' + latest_update + '/pipeline.pkl')
L2SGD = joblib.load('binaries/' + latest_update + '/L2SGD.pkl')
elasticNet = joblib.load('binaries/' + latest_update + '/elasticNet.pkl')
L1SVC = joblib.load('binaries/' + latest_update + '/L1SVC.pkl')
MNB = joblib.load('binaries/' + latest_update + '/MNB.pkl')
L1SGD = joblib.load('binaries/' + latest_update + '/L1SGD.pkl')
passiveAggressive = joblib.load('binaries/' + latest_update + '/passiveAggressive.pkl')
randomForest = joblib.load('binaries/' + latest_update + '/randomForest.pkl')
adaBoost = joblib.load('binaries/' + latest_update + '/adaBoost.pkl')
bagging = joblib.load('binaries/' + latest_update + '/bagging.pkl')

predictors = {'ridge':ridge, 'logit':logit, 'nearestCentroid':nearestCentroid, 'L2SVC':L2SVC, 'BNB':BNB, 'knn':knn, 'perceptron':perceptron, 'pipeline':pipeline, 'L2SGD':L2SGD, 'elasticNet':elasticNet, 'L1SVC':L1SVC, 'MNB':MNB, 'L1SGD':L1SGD, 'passiveAggressive':passiveAggressive, 'randomForest':randomForest, 'adaBoost':adaBoost, 'bagging':bagging}

predictionList = []
for i in range(0, len(solicitations)):
    subDict = {}
    subDict['url'] = solicitations[i].document_status_final['parent_url']
    subDict['text'] = solicitations[i].doc_text
    subDict['predictions'] = {}
    predictionList.append(subDict)

predictionText = []
for item in predictionList:
    predictionText.append(item['text'])

print('vectorizing text')
vec = joblib.load('binaries/' + latest_update + '/vectorizer.pkl')

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

dataHandling.writeJson(formatted.finalOutput, 'predictions_' + (datetime.today() - timedelta(days=1)).strftime("%Y%m%d") + '.json')
"""
from pymongo import MongoClient
from datetime import datetime, timedelta

client = MongoClient()
db = client.FBO_Test
collection = db.test_collection
insertion = {'date':(datetime.today() - timedelta(days=1)).strftime("%Y%m%d"), 'report' = formatted.finalOutput}
collection.insert_one(insertion)
"""