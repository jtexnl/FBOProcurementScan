# FBO Procurement Scan
### A program for predicting which solicitations are likely to not be in conformance with Section 508 requirements

##### About Section 508 Procurement Review
Section 508 of the Rehabilitation Act of 1973 mandates that all government technology be made accessible to persons with disabilities. For example, government websites are supposed to be built to be compatible with screen-reading software so that the page's information and functionalities are accessible to the visually-impaired.

The government contracts with private companies for services and projects quite often. When the government enters into a contract, it is important that the requirements relating to Section 508 be clearly spelled out. Ultimately, the government is responsible for ensuring that new products are accessible, so companies can only be held accountable for not producing accessible products if the accessibility requirements were explicitly defined in the legal documents relating to the contract. These legal documents are listed publicly on [FedBizOpps]('https://fbo.gov'), a GSA website that serves as one of the central points where contracting opportunities are listed (it is not the only source, but it is the largest). 

A part of GSA's Section 508 Program is to assist agencies in performing solicitation review, a process by which experts in Section 508 law review new procurement documents as they are posted to FedBizOpps for conformance with Section 508 requirements. This is a daunting task, as FedBizOpps often sees in excess of 500 new opportunities posted every day, including even weekends and holidays. 

This program was written as an aide to those performing this soliciation review process. GSA has been centrally reviewing a sample of solicitations (an average of 20-40 per month) since 2009. This left us with a large set of data: past procurements and their level of conformance (graded as 'red' for entirely non conformant, 'green' for fully conformant, and 'yellow' for partially conformant). We began by collecting documents from the graded procurements, parsing them into raw text, then applying several Natural Language Processing (NLP) techniques to tokenize, lemmatize (revert a word to its basic form), and vectorize the text. With the vectorized text, we trained several well-known machine learning algorithms to look for patterns associated with green vs. yellow vs. red procurements, then scored the models using [k-folds cross validation]('https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation'). 

The final piece was a module for pulling new solicitations from FedBizOpps, parsing/vectorizing their documents, and predicting their level of conformance based on the outputs of the trained machine learning algorithms. 

The outputs here should **NOT** be taken as an official evaluation of the conformance level of a document. This project is still in its alpha stage, and even when the program is in a production phase, machine learning can only take one so far in evaluating procurements. It is still vital that procurements be reviewed by Section 508 SMEs in order to make a definitive determination. The purpose of this tool is to act as a filter, aiding 508 SMEs in finding the solicitations most likely to not be in conformance with Section 508, so that they can be further evaluated and so that amendments can be made before the solicitation closes. 

##### System Requirements
The bulk of this program is written in Python 3.4.3. It has not been tested in a Python 2.x environment, though it would likely work with some minor adjustments. The path variables in the program are written with a Unix/Linux operating system in mind, so some edits will be needed to run this program on a Windows machine (notably changning the ```/``` characters in paths to ```\```).

For document parsing, the program uses Textract, a Python library for parsing most types of documents into plain text. As of this writing, Textract is not optimized for use with Python 3.4.3, so you will need to make some edits to the base libraries to run this tool. Instructions to follow shortly.

You will also need to have Node.js and npm installed on the machine that is running the software. Further instructions on this can be found in ```scripts/pull```.


# Setup

Setting up your environment to use this tool is not exactly straightforward, so I'm going to run through all of the steps required. For this example, I'm going to be documenting the steps I went through to set up this tool to run on an Ubuntu AWS EC2 Instance. Because of the size of the data being used, a micro instance is not recommended: you will need something with a larger amount of virtual memory. 

I do not recommend attempting to set this tool up in a Windows environment. Several of the libraries are not optimized for the Windows OS and may require expensive programs like Visual Studio, in addition to complicated changes to the path variables. Setup is much more straightforward in a Unix-like environment. It may not be impossible for Windows, but I'm not going to go through the steps for doing that here. 

If you would like to run this on your own machine, I'd recommend setting up a virtual environment using something like ```venv```. The setup here is for a server dedicated to this tool, so I won't be installing in a virtual environment.

First and foremost, once you're in your Ubuntu instance, update apt-get to make sure it's able to fetch the packages you're going to try and download later: 
```sudo apt-get update```

## Quick Setup
If your environment is similar enough to the one I was working in, you can try to run the whole setup procedure by running my setup script. To do that, simply try:
```bash setup.sh```
There may be a couple of steps where you're asked to type 'yes' before proceeding, so pay attention to what's happening on the console as it runs through the installation script.
If you notice warnings or failures in the console output, go through the manual setup below.

## Manual Setup

### Python
You will need Python 3.4.3 or later to run this tool. Most systems come with Python 2.7x pre-installed, so you'll need to download Python 3.4.3+. To do this, run:
```sudo apt-get install python3```
As of this writing, the default install will be Python 3.5.2, which is fine for our purposes. With Python in place, you'll need to install your necessary Python libraries. To do this, you should download the Python package manager Pip. Some versions of Python ship with Pip, but you'll want to make sure that you have pip for Python 3; the default is for Python 2, so unless you download the correct version, the libraries will download to the root of Python 2 instead of Python 3, making it so that you can't import the libaries into Python 3. To get Pip for Python 3, type: 
```sudo apt-get install --upgrade python3-pip```
Once Pip is set up correctly, install the basic Python libraries you need by typing: 
```sudo pip3 install -r --upgrade requirements.txt```
You can also manually go through each library in requirements.txt to do the install if something breaks here, but it should hopefully work. 

### NLTK and SpaCy
There are two libraries used for Natural Language Processing in this tool, but both of them require an additional installation step that involves downloading their data sets (including things like stopwords and text corpora). To do this, run the following commands:
```sudo python3 -m nltk.downloader all```
```sudo python3 -m spacy.en.download all```
These commands will take some time to run and will likely require you to confirm running the installation (as the installation will take up a good amount of disk space).

### Curl and WGET
Curl and WGET will be used by the script to fetch documents from websites. Your system likely already has these installed, but to be sure, enter:
```sudo apt-get install curl```
```sudo apt-get install wget```

### Textract
Textract is a utility that pulls data from documents and parses it into text. It has utilities for all types of documents and will at least try to parse everyting, so it has a LOT of dependencies (think OCR readers). This is the most fragile step, and you may need to do some troubleshooting on StackOverflow to make this work for your particular environment. Importantly, although textract is a Python package that you download with Pip, it expects all of the dependencies to already be in place, so you have to follow this download order or the installation will fail. This is what I did for my environment: 

1. Install the developer libraries
```sudo apt-get install python-dev libxml2-dev libxslt1-dev libjpeg-dev ```
2. Install the CL utilities that Textract relies on
```sudo apt-get install antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox```
3. Install a couple of other dev libraries (I don't know why these aren't in step 1, but this is what worked for me)
```sudo apt-get install lib32z1-dev zlib1g-dev```
4. Install lxml (necessary for parsing a lot of Microsoft document types)
```sudo apt-get install lxml```
5. Finally, instal Textract
```sudo pip3 install -U textract```

### Conclusion
That's it for your overall development setup, but also be sure that you correctly install the node dependencies in the ```pull/``` folder. Instructions are in the readme there. 