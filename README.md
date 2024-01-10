# Disaster-Response-Pipeline-Project-

## Table of Contents
1. [Overview](#overview)
2. [Project Overview](#project-overview)
3. [Getting Started](#getting-started)
4. [Installation](#installation)
5. [Running the Program](#running-the-program)
6. [Additional Resources](#additional-resources)
7. [License](#license)
8. [Credits](#credits)

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
git clone https://github.com/canaveensetia/udacity-disaster-response-pipeline.git
```


## Running the Program
1. Execute the ETL pipeline to clean and store processed data in the database:
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
```
2. Run the ML pipeline to load data, train the classifier, and save the model:
```
python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
```
3. Launch the web app by running the following command in the app's directory:
```
python run.py
```
4. Access the web app at http://0.0.0.0:3000/.


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

