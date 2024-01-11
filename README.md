# Disaster-Response-Pipeline-Project-

## Table of Contents
1. [Overview](#overview)
2. [Project Overview](#project-overview)
3. [Getting Started](#getting-started)
4. [Installation](#installation)
5. [Running the Program](#running-the-program)
6. [Files Description](#files-description)
7. [Additional Resources](#additional-resources)
8. [License](#license)
9. [Credits](#credits)

## Overview
Welcome to the Disaster Response Project! As part of the Udacity Data Science Nanodegree Program in collaboration with Figure Eight, this project focuses on developing a Natural Language Processing (NLP) model for real-time categorization of messages during disaster events.

## Project Overview
### Data Processing (ETL Pipeline)
The initial phase involves processing data through an Extract, Transform, Load (ETL) pipeline. This includes:

* Extracting data from the source.
* Cleaning and organizing the data.
* Storing the processed data in a SQLite database.
  
###  Machine Learning Pipeline
The heart of the project lies in building a robust machine learning pipeline. Key steps include:

* Constructing a pipeline to facilitate the training of a classifier for text message categorization.
* Utilizing the pipeline to train the model, enabling it to classify messages into various predefined categories.
  
### Web App
To make the model results accessible in real-time, a web application is deployed.

## Getting Started

### Dependencies
Ensure you have the following dependencies installed:

* Python 3.5+
* Key Libraries: NumPy, SciPy, Pandas, Scikit-Learn
* Natural Language Processing: NLTK
* SQLite Database: SQLalchemy
* Model Handling: Pickle
* Web App Development: Flask, Plotly


## Installation
Clone the repository:
```
git clone https://github.com/Turkii17/Disaster-Response-Pipeline-Project-.git
```


## Running the Program
1. Execute the ETL pipeline to clean and store processed data in the database:
```
python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db
```
2. Run the ML pipeline to load data, train the classifier, and save the model:
```
python train_classifier.py disaster_response_db.db classifier.pkl
```
3. Launch the web app by running the following command in the app's directory:
```
python run.py
```
4. Access the web app at http://0.0.0.0:3000/.

## Files Description
### Web App Templates:
* `go.html`: an html file for the web app
* `master.html`: another html file for the web app
  
### ETL Pipeline Script:

* `process_data.py`
Description: Handles Extract, Transform, and Load (ETL) processes.
Performs data cleaning, feature extraction, and stores data in a SQLite database (disaster_response_db.db).

### Machine Learning Pipeline Script:

* `train_classifier.py`
Description: Implements a machine learning pipeline.
Loads data, trains a classifier, and saves the trained model as a .pkl file for future use.

### Web App Launcher Script:

* `run.py`
Description: Launches the Flask web app used to classify disaster messages.

## Additional Resources
For a deeper understanding of the model, explore the provided Jupyter notebooks in the **'data'** and **'models'** folders:
* **ETL Preparation Notebook**: Comprehensive insights into the implemented ETL pipeline.
* **ML Pipeline Preparation Notebook**: Detailed exploration of the NLP and Scikit-Learn powered ML pipeline.
  
Use the ML Pipeline Preparation Notebook for retraining or tuning the model through the dedicated Grid Search section.

## License
MIT License

## Credits
* **Udacity** for providing an outstanding Data Science Nanodegree Program.
* **Figure Eight** for generously providing the relevant dataset to train the model.

