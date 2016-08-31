from datetime import datetime, timedelta
import os
import requests as rq 
from spacy.en import English
import dataHandling

#Class for tokenizing and cleaning the data pulled from FBO

fbo_base_url = 'https://www.fbo.gov'
parser = English()
STOPLIST = set(stopwords.words('english') + ['n\'t', '\'s', '\'m', 'ca'] + list(ENGLISH_STOP_WORDS))
SYMBOLS = ' '.join(string.punctuation).split(' ') + ['-----', '---', '...', '“', '”', '\'ve']

class newSolicitation():
    def __init__(self, index):
        self.dataFile = newSols
        self.url = newSols[index]['listing_url']
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