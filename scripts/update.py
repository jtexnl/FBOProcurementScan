#This script should be run periodically as more data is added to update the classifier binaries. 
#Be sure to consult the readme about folder structure and naming conventions, as they're essential to having this program run correctly.
#Remember, this script takes a long time to run, especially if you include the entire reject list as a filter
#Without the reject list, it's probably one hour; with the reject list, closer to 12.
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

def loadData():
    data = classes.dataDict()
    dataDict = {}
    dataDict['DESCR'] = None 
    dataDict['data'] = data.contents
    dataDict['target_names'] = ['RED', 'YELLOW', 'GREEN']
    dataDict['target'] = data.grades
    dataDict['description'] = 'dataset of graded solicitations from 2009-2012'
    return dataDict

if __name__ == '__main__':
    dataDict = loadData()

    data_train, data_test = dataHandling.makeTrainTest(dataDict, .2)
    spread = np.mean(data_train['target']) - np.mean(data_test['target'])
    while abs(spread) >= .05:
        data_train, data_test = dataHandling.makeTrainTest(dataDict, .2)

    y_train, y_test = data_train['target'], data_test['target']
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train = vectorizer.fit_transform(data_train['data'])
    X_test = vectorizer.transform(data_test['data'])

    categories = dataDict['target_names']
    feature_names = vectorizer.get_feature_names()
    feature_names = np.asarray(feature_names)

    ridge = RidgeClassifier(tol=1e-2, solver="lsqr")
    perceptron = Perceptron(n_iter=50)
    passiveAggressive = PassiveAggressiveClassifier(n_iter=50)
    knn = KNeighborsClassifier(n_neighbors=10)
    randomForest = RandomForestClassifier(n_estimators=100)
    L1SVC = LinearSVC(loss='l2', penalty='L1', dual=False, tol=1e-3)
    L2SVC = LinearSVC(loss='l2', penalty='L2', dual=False, tol=1e-3)
    L1SGD = SGDClassifier(alpha=.0001, n_iter=50, penalty='L1')
    L2SGD = SGDClassifier(alpha=.0001, n_iter=50, penalty='L2')
    elasticNet = SGDClassifier(alpha=.0001, n_iter=500, penalty="elasticnet")
    nearestCentroid = NearestCentroid()
    MNB = MultinomialNB(alpha=.01)
    BNB = BernoulliNB(alpha=.01)
    logit = LogisticRegression()
    pipeline = Pipeline([
      ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
      ('classification', LinearSVC())
    ])

    classifiers = [ridge, perceptron, passiveAggressive, knn, randomForest, L1SVC, L2SVC, L1SGD, L2SGD, elasticNet, nearestCentroid, MNB, BNB, logit, pipeline]
    classifier_names = ['ridge', 'perceptron', 'passiveAggressive', 'knn', 'randomForest', 'L1SVC', 'L2SVC', 'L1SGD', 'L2SGD', 'elasticNet', 'nearestCentroid', 'MNB', 'BNB', 'logit', 'pipeline']