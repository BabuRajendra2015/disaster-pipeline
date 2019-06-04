# disaster-pipeline
# DataScience-NanoDegree

Disaster Response Pipelines

# Table of Contents

1. Installation
2. Project summary
3. Dataset
4. File Descriptions

#Installation

The code runs on Anaconda distribution of Python version 3.*. The project contains three folders data, app and models. The data folder contains process_data.py, disaster_categories.csv, disaster_messages.csv and DisasterResponse.db. The models folder contains train_classifier.py. The app folder contains templates folder which contains app, go.html and master.html. The app folder also contains run.py which is the key file for launching web app.


# Project summary:

This project is about grouping the disaster messages under various categories like water, medicine etc. pandas data frames is used for data processing and ML pipelines will be used for processing the text. After grouping the messages, visualizations will be created and launched through a web app. The project files should be launched as below on the python terminal

python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
python run.py

The web app can be run as below

In a seperate terminal, type in env|grep WORK, and get the spacid and space domain. Launch the app in the web brwoser as mentioned below
https://SPACEID-3001.SPACEDOMAIN



# Dataset: 
The disaster_messages.csv and disaster_categories.csv contains the disaster messages to be processed and categorized. 


# File Descriptions
 - app -- template
       - master.html - main page of web app
       - go.html - classification result page of web app
- app -- run.py - Flask file that runs app

- data 
- disaster_categories.csv – file containing the categories  
- disaster_messages.csv – file containing disaster messages to be 
   categorized
- process_data.py – file containing data loaded code
- DisasterResponse.db - database to save the cleaned and 
 categorized data	

- models
- train_classifier.py – file containing ML pipeline
- classifier.pkl – model is saved in this pickle file
