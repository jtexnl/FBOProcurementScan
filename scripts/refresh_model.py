import kfolds
from baseModel import classifiers, classifier_names, benchmark
from loadData import loadData
import dataHandling
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidVectorizer
from datetime import datetime

def writeJson(inputData, fileName):
    with open(fileName, 'w+') as outfile:
        json.dump(inputData, outfile, indent = 4)

#get the date
today = datetime.today().strftime('%Y%m%d')
#load the data. This step will take a while
dataDict = loadData.loadData()
#split the dataset; 5 pieces is the default, but not the rule necessarily
splits = dataHandling.kfolds_split(dataDict, 5)
#get the accuracy scores for each of the classifiers and the vectorizer
accuracyDict = kfolds.test_model_accuracy(splits, dataDict)

#write the accuracy dict and the dataDict so that they will persist for future use
writeJson(accuracyDict, 'accuracyDict_' + today + '.pkl')
joblib.dump(dataDict, 'dataDump_' + today + '.pkl')

#vectorize the data and train the classifiers
y = dataDict['target']
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
joblib.dump(vectorizer, 'vectorizer.pkl')
X = vectorizer.fit_transform(dataDict['data'])

#pickle the classifiers
for i, clf in enumerate(classifiers):
    clf.fit(X,y)
    joblib.dump(clf, classifier_names[i] + '.pkl')