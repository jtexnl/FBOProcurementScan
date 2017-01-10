#This script should be run periodically as more data is added to update the classifier binaries. 
#Be sure to consult the readme about folder structure and naming conventions, as they're essential to having this program run correctly.

from sklearn.externals import joblib
import numpy as np
import dataHandling
import classes
from algorithms import classifiers, classifier_names
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import os

if __name__ == '__main__':
    #load the raw data files into a properly-formatted dictionary. This will take a lot of time.
    dataDict = dataHandling.loadData()
    #For the time being, we are using the binary classifier model, so we will need to run the convert_to_binary function to make this dictionary binary
    dataDict = dataHandling.convert_to_binary(dataDict)
    #get the date
    today = datetime.datetime.today().strftime('%Y%m%d')
    #run k-folds cross-validation to derive accuracy scores
    #split the dataset; 5 pieces is the default, but not the rule necessarily
    splits = dataHandling.kfolds_split(dataDict, 5)
    accuracyDict = dataHandling.test_model_accuracy(splits, dataDict)
    #write the scores to a new accuracy json and pickle the dataset for persistence
    dataHandling.writeJson(accuracyDict, 'accuracy_scores/accuracyDict_' + today + '.json')
    os.mkdir('binaries/' + today)
    joblib.dump(dataDict, 'binaries/' + today + '/dataDump.pkl')
    #vectorize the whole dataset and train the algorithms on the full dataset
    y = dataDict['target']
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    joblib.dump(vectorizer, 'binaries/' + today + '/vectorizer.pkl')
    X = vectorizer.fit_transform(dataDict['data'])
    #dump the algorithms and vectorizer for re-use in prediction
    for i, clf in enumerate(classifiers):
        clf.fit(X,y)
        joblib.dump(clf, 'binaries/' + today + '/' + classifier_names[i] + '.pkl')
    #create a .txt file in the binaries folder containing the date of this latest update. 
    #Every time a successful update is run, this file will reflect the latest date for which binaries are available, and only those binaries will be opened for prediction
    with open('latest_update.txt', 'w+') as outfile:
        outfile.write(today)
