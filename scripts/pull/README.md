# Pull
### Module for pulling new solicitations from FedBizOpps
FedBizOpps publishes a daily digest of all new solicitations posted on the previous day. This data is posted as XML; 18F [created an API]('https://github.com/18F/fbopen') to convert these data dumps into more readily-searchable JSON data. We've borrowed some of their code and repurposed it for this tool. While running the program will automatically call these scripts and pull the json data, you can also run the code here. 

First, though, be sure that you have all of the dependencies installed. For this, you'll need to have node.js and npm on your machine. Once you have these in place, type ```npm install -r requirements.txt``` to install all of the needed node libraries for running this set of scripts. 

Once your dependencies are in place, type ```bash fbo-nightly.sh```. If you don't already have a directory called ```workfiles```, it will create one for you, and the json output will be stored in this directory. The scripts will actually create three files every time it is run: 
* ```FBOFeedYYYYMMDD.txt``` is the raw XML as it was pulled from FBO
* ```FBOFeedYYYYMMDD.txt.json``` is a json conversion of the raw XML
* ```prepped_notices.YYYYMMDD.json``` is the final json file that will be used by the prediction program