#This script should be run periodically as more data is added to update the classifier binaries. 
#Be sure to consult the readme about folder structure and naming conventions, as they're essential to having this program run correctly.

from sklearn.externals import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import density
from sklearn import metrics
import dataHandling
import classes
from baseModel import classifiers, classifier_names

if __name__ == '__main__':
    #load the raw data files into a properly-formatted dictionary
    dataDict = dataHandling.loadData()
    #get the date
    today = datetime.today().strftime('%Y%m%d')
    #run k-folds cross-validation to derive accuracy scores
    #split the dataset; 5 pieces is the default, but not the rule necessarily
    splits = dataHandling.kfolds_split(dataDict, 5)
    accuracyDict = dataHandling.test_model_accuracy(splits, dataDict)
    #write the scores to a new accuracy json and pickle the dataset for persistence
    writeJson(accuracyDict, 'accuracyDict_' + today + '.pkl')
    joblib.dump(dataDict, 'dataDump_' + today + '.pkl')
    #vectorize the whole dataset and train the algorithms on the full dataset
    y = dataDict['target']
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    X = vectorizer.fit_transform(dataDict['data'])
    #dump the algorithms and vectorizer for re-use in prediction
    for i, clf in enumerate(classifiers):
        clf.fit(X,y)
        joblib.dump(clf, 'binaries/' + classifier_names[i] + '.pkl')
