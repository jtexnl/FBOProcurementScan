#Note: these functions can probably be moved into the loadData.py script for compactness
import re
import string
from spacy.en import English
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

#The reject list was originally constructed as a way of filtering out words found in the tokenized solicitations that didn't mean anything. 
#Some testing has shown that the list doesn't vastly improve accuracy, so I'm commenting it out here, as it otherwise makes loading data very processor-intensive and slow
#
rejectList = []

parser = English()

STOPLIST = set(stopwords.words('english') + ['n\'t', '\'s', '\'m', 'ca'] + list(ENGLISH_STOP_WORDS))
SYMBOLS = ' '.join(string.punctuation).split(' ') + ['-----', '---', '...', '“', '”', '\'ve']

def removeStrings(string, removeList):
    strList = string.split(' ')
    newList = []
    for item in strList:
        if not item in removeList:
            newList.append(item)
    newString = ' '.join(newList)
    return newString

def tokenizeText(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip())
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")
    tokenStrings = [str(tok) for tok in tokens]
    for string in tokenStrings:
        string.replace(' ', '')
    newText = ' '.join(tokenStrings)
    return newText

def cleanUpText(inputText, rejectList):
    noRejects = removeStrings(inputText, rejectList)
    tokenized = tokenizeText(noRejects)
    return tokenized