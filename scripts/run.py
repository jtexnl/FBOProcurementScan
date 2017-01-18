##########################################################################################################
# A Script for collecting solicitations and their associated documents for a given day                   #
# Written by John Tindel, using code contributed by 18F's FBOpen project                                 #
##########################################################################################################

##########################################################################################################
#TODO: Currently this script will only pull the solicitations from yesterday. Adding an option to pull from a given day would be useful
##########################################################################################################

import classes
import os
from spacy.en import English
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import numpy as np
import json
from datetime import datetime, timedelta

#Run the bash script to pull the most recent FBO Data Dump
#This script runs a node utility that was part of the FBOpen API. The code for it was written by 18F and is being 
#used thanks to their decision to make it open-source. The utility will produce three documents: one is a .txt file
#containing xml, one is a json conversion of that xml file, and the other is a formatted json file named prepped_notices.YYYYMMDD.json
os.system('bash pull/fbo-nightly.sh')

#Initialize the dailySolicitationListing class to get the list of new solicitations. This uses the prepped_notices.YYYYMMDD.json file from
#the previous step and produces an instance of a class containing information such as the url, some metadata, etc.
#for a full list of class attributes, see the code for this in classes.py
print('Loading raw data')
rawData = classes.dailySolicitationListing()

#Create a new directory where the documents will be stored until parsed. This directory will be deleted at the end
os.mkdir('temp_test')
#Move into the temporary directory
os.chdir('temp_test')
print('scraping and parsing solicitation documents')
#Create an empty list of solicitations 
solicitations = []
#Iterate through every element in rawData.raw (a list). This section of code will take the longest, but is still only about 5 minutes ususally
for i in range(0, len(rawData.raw)):
    print('scraping solicitation number ' + str(i) + ' for documents')
    #For each item, instatiate the solicitation_documents class, which will pull and parse all of the 
    #documents found on each page. In the end, you will have a list called 'solicitations', and each element in 
    #the list will be an instance of the solicitation_documents class. Look at the code for solicitation_documents
    #to see the available attributes of this class.
    solicitations.append(classes.solicitation_documents(rawData.raw[i]))

#Once you have scraped all of the solicitation websites for documents, move back to the scripts directory
os.chdir('..')
#Delete the temporary directory containing all of the raw documents (they will have formats like .doc, .pdf, etc.)
os.system('rm -r temp_test')
#Open the latest_update.txt file, which will contain the date that update.py was last run in YYYMMDD format
#This will be used to find the latest available vectorizer and algorithm pickles
##########################################################################################################
#TODO: This might be easier if we use an environment variable, so maybe look into that?###################
##########################################################################################################
latest_update = open('latest_update.txt').read()
#Using the latest_update variable, load the binaries (pickles) from the folders containing the latest-updated binaries
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

#Since typing the algorithm names themselves will only display the model parameters, this dictionary will
#allow you to call the models as well as us the string representation of their names, which will be important in reading the prediction output
predictors = {'ridge':ridge, 'logit':logit, 'nearestCentroid':nearestCentroid, 'L2SVC':L2SVC, 'BNB':BNB, 'knn':knn, 'perceptron':perceptron, 'pipeline':pipeline, 'L2SGD':L2SGD, 'elasticNet':elasticNet, 'L1SVC':L1SVC, 'MNB':MNB, 'L1SGD':L1SGD, 'passiveAggressive':passiveAggressive, 'randomForest':randomForest, 'adaBoost':adaBoost, 'bagging':bagging}

#Create a blank list called 'predictionList'
predictionList = []
#Go trough every element in the list of solicitations (remember, each element is an instance of the 'solicitation_documents' class)
for i in range(0, len(solicitations)):
    #create an empty dictionary
    subDict = {}
    #assign the 'url' key to the value of 'parent_url' from the document_status_final attribute of the current solicitation_documents instance
    subDict['url'] = solicitations[i].document_status_final['parent_url']
    #assign the 'text' key to the doc_text attribute of the current solicitation. This is the parsed/lemmatized text with all of the punctuation removed
    subDict['text'] = solicitations[i].doc_text
    #create another blank dictionary that will hold the prediction output. It will be filled in a later step
    subDict['predictions'] = {}
    #append the contents of the dictionary (urls, raw text, and an empty dictionary for predictions) to the predictionList
    predictionList.append(subDict)

#We now have a list of dictionaries containing the raw text and the urls. urls will be used as an identifier going forward

#Our next step is to isolate the text from all of the solicitations into a new list, which will become our feature vector
#To do this, create an empty list called 'predictionText'
predictionText = []
#Iterate through every item in predictionList
for item in predictionList:
    #and append the value associated with 'text' to the list
    predictionText.append(item['text'])

#We now have a list of strings representing the list of all documents from our scraping run. 
#This list will have empty elements for solicitations that didn't have documents or which failed to parse, but it's important to leave the empty
#elements in place for now, as they are necessary for indexing. Empty elements should have a 'red' prediction

print('vectorizing text')
#load the vectorizer from the latest update. It is vitally important that you use the vectorizer that was used for training the algorithms, as the
#algorithms will expect a matrix to be passed in that is exactly the same shape as their training data. Making a new vectorizer will not work, as it
#will produce a differently-shaped matrix and the algorithms will throw an error
vec = joblib.load('binaries/' + latest_update + '/vectorizer.pkl')
#After loading the vectorizer, transform the text list into a sparse matrix using the vectorizer.
predictionVector = vec.transform(predictionText)

print('generating predictions')
#Now, we'll iterate through the dictionary of predictors and predictor names
for name, predictor in predictors.items():
    #for each predictor, use the .predict() function on the prediction vector. This will return a list of 0's (corresponding to 'red') and 1's ('green') 
    results = predictor.predict(predictionVector)
    #go through each element in the resulting list of 1's and 0's
    for i in range(0, len(results)):
        #and assign the result from that index to the dictionary for that element in predictionList. The key will be the algorithm name, and the value will be the predction
        predictionList[i]['predictions'][name] = results[i]

#Now we need to get an average prediction
##########################################################################################################
#TODO: I'm not sure if this section is necessary. There should be a weighting system, which I think comes in under formattedPredictionOutput. 
#Someone should investigate if this step is actually necessary.
##########################################################################################################
for item in predictionList:
    values = list(item['predictions'].values())
    item['avgPred'] = np.mean(values)

#Now that we have the predictions, we need to format them correctly and calculate the weighted averages
#start by creating a blank dictionary called 'raw', which will be fed into the 'formattedPredictionOutput' class
##########################################################################################################
#TODO: I am fairly sure code here can be consolidated or rewritten in a cleaner fashion; a lot of this was
#written this way just because I needed to get it to work at the time. Some of this is repetitive, so a 
#possible task for making this tool better is to rewrite this whole section
##########################################################################################################
raw = {}
#iterate through every element in predictionList
for i in predictionList:
    #select the 'relevant' (ergo 'rel') element of the dictionary, which is 'predictions', 
    #a dictionary containing the predictions each algorithm made for the solicitation in question
    rel = i['predictions']
    #enter the information from the 'predictions' dictionary into the 'raw' dictionary, which will feed 'formattedPredictionOutput()'
    for key, value in rel.items():
        if not key in raw:
            raw[key] = [value]
        else:
            raw[key].append(value)

#pass the two needed elements (raw, the dictionary of predictions by algorithm; and solicitations, the list of solicitations pulled)
#into the formattedPredictionOutput class, which will weigh the predictions by their accuracy and produce a final predicted score 
formatted = classes.formattedPredictionOutput(raw, solicitations)

#Temporarily, we are writing this to json on the EC2 instance. Once we decide on a schema for Mongo, we can delete this 
#step and move this all into a DB insert
dataHandling.writeJson(formatted.finalOutput, 'predictions_' + (datetime.today() - timedelta(days=1)).strftime("%Y%m%d") + '.json')
#The code below shows how you could go about doing the DB insertion. It's currently not operational until we agree on a schema.
"""
from pymongo import MongoClient
from datetime import datetime, timedelta

client = MongoClient()
db = client.FBO_Test
collection = db.test_collection
insertion = {'date':(datetime.today() - timedelta(days=1)).strftime("%Y%m%d"), 'report' = formatted.finalOutput}
collection.insert_one(insertion)
"""