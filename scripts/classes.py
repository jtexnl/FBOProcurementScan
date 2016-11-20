from datetime import datetime, timedelta
import json
import dataHandling
import requests as rq 
from pyquery import PyQuery as pq
import wget
from subprocess import call
import os
from urllib.parse import urljoin
import glob
import textract
import re
import string
from spacy.en import English
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

#The daily solicitation listing class is used to open the files from 
class dailySolicitationListing():
    
    def __init__(self):
        self.date = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")
        self.fileName = 'workfiles/prepped_notices.' + str(self.date) + '.json'
        self.raw = self.open_and_parse(self.fileName)
        self.urls = self.make_url_list(self.raw)

    #This function opens the json file produced by the FBOpen API scripts and parses it into a Python list of dictionaries
    def open_and_parse(self, fileName):
        data = []
        with open(fileName) as infile:
            for line in infile:
                data.append(json.loads(line))
        return data

    #This function returns a list of urls from the new load of solicitations. The urls will be used to collect documents
    def make_url_list(self, raw):
        urls = []
        for i in raw:
            urls.append(i['listing_url'])
        return urls

#This class uses the dailySolicitationListing class and scrapes fedBizOpps for the documents. 
#It then processes the documents and returns formatted data ready to be vectorized for classification
fbo_base_url = 'https://www.fbo.gov'
parser = English()
STOPLIST = set(stopwords.words('english') + ['n\'t', '\'s', '\'m', 'ca'] + list(ENGLISH_STOP_WORDS))
SYMBOLS = ' '.join(string.punctuation).split(' ') + ['-----', '---', '...', '“', '”', '\'ve']

class newSolicitation():

    def __init__(self, inputList, index):
        self.dataFile = inputList[index]
        self.url = inputList[index]['listing_url']
        self.agency = inputList[index]['agency']
        self.attachments = self.collect_link_attrs(self.url)
        self.rawContents = self.parseAttachments(self.attachments)
        self.tokenized = self.tokenize(self.rawContents)

    def collect_link_attrs(self, url):
        doc = pq(url)
        attachments = []
        count = 0
        for div in doc('#dnf_class_values_procurement_notice__packages__widget > div.subform.readonly.subreadonly'):
            try:
                d = pq(div)
                link_tag = d.find('div.file')('a') or d.find('div')('a')
                attachments.append(urljoin(fbo_base_url, link_tag.attr('href')))
            except:
                continue
        if not attachments:  # keep looking
            addl_info_link = doc('div#dnf_class_values_procurement_notice__additional_info_link__widget')('a')
            if addl_info_link:
                attachments.append(addl_info_link.attr('href'))
        return attachments

    def parseAttachments(self, attachments):
        os.mkdir('temp')
        os.chdir('temp')
        for attachment in self.attachments:
            try:
                filename = wget.download(attachment)
                call(["curl", "-o ", filename, " '", attachment, "'"])
            except SyntaxError:
                continue
            except:
                continue
        if 'description' in self.dataFile.keys():
            final = self.dataFile['description']
        else:
            final = ''
        if len(os.listdir()) > 0:
            for filename in os.listdir():
                temp = str(textract.process(filename))
                final = final + temp
        os.chdir('..')
        if 'temp' in os.listdir():
            os.system('rm -r temp')
        return final
        
    def tokenize(self, text):
        cleaned = dataHandling.transform_for_classifier(text)
        tokens = parser(cleaned)
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
        tokenized = ' '.join(tokenStrings)
        return tokenized

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
            subDict['url'] = self.solicitationList[i].url
            subDict['agency'] = self.solicitationList[i].agency
            subDict['predictions'] = self.gradesBySolicitation[i]
            finalOutput.append(subDict)
        return finalOutput