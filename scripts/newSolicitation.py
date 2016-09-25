from datetime import datetime, timedelta
import os
import requests as rq 
from spacy.en import English
import dataHandling
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from urllib.parse import urljoin
import string
import glob
import textract
import re
import dataHandling
from sklearn.externals import joblib
from pyquery import PyQuery as pq
import wget
from subprocess import call

#Class for tokenizing and cleaning the data pulled from FBO

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

class apiCall():

    def __init__(self):
        self.date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        self.url = self.build_url(self.date)
        self.data = self.request_data(self.url)
        self.solURLs = self.get_sol_urls(self.data)

    def build_url(self, date):
        api_key = 'bmPVCo4hkTV2SWMWjXEvx7XYl6gYc77BpGlhseXq'
        base_url = 'https://api.data.gov/gsa/fbopen/v0/opps?q=posted_dt:'
        url = base_url + self.date + 'T00:00:00Z&data_source=FBO&limit=500&api_key=' + api_key
        return url

    def request_data(self, url):
        res = rq.get(self.url)
        data = res.json()['docs']
        return data

    def get_sol_urls(self, data):
        solURLS = []
        for i in range(0, len(self.data)):
            try:
                solURLS.append(self.data[i]['listing_url'])
            except KeyError:
                continue
        return solURLS
