# Disaster-Response-Pipeline-Project-

## Table of Contents
1. [Overview](#overview)
2. [Community Impact and Disaster Response](#community-impact-and-disaster-response)
3. [Project Overview](#project-overview)
4. [Getting Started](#getting-started)
5. [Installation](#installation)
6. [Running the Program](#running-the-program)
7. [Files Description](#files-description)
8. [Additional Resources](#additional-resources)
9. [License](#license)
10. [Credits](#credits)

## Overview
Welcome to the Disaster Response Project! As part of the Udacity Data Science Nanodegree Program in collaboration with Figure Eight, this project focuses on developing a Natural Language Processing (NLP) model for real-time categorization of messages during disaster events.

## Community Impact and Disaster Response
The Disaster Response Project plays a pivotal role in enhancing community resilience during times of disaster. By leveraging Natural Language Processing (NLP) and machine learning, this application offers significant benefits to both individuals and organizations involved in disaster response efforts.

1. **Rapid Categorization of Messages**:
**Issue**: During a disaster, the influx of messages can overwhelm response teams.
**Solution**: The NLP model swiftly categorizes incoming messages, enabling rapid identification of critical information.
2. **Targeted Resource Allocation**:
**Issue**: Limited resources must be allocated efficiently to areas in need.
**Solution**: By classifying messages into specific categories (e.g., medical assistance, food supply), organizations can prioritize and allocate resources effectively.
3. **Enhanced Situational Awareness**:
**Issue**: Understanding the evolving situation is challenging in the midst of a disaster.
**Solution**: The application provides real-time insights, allowing organizations to adapt their response strategies based on the most recent information.
4. **Optimized Communication**:
**Issue**: Clear communication is crucial but can be hindered by information overload.
**Solution**: The model filters and categorizes messages, streamlining communication channels and ensuring that relevant information reaches the right teams.
5. **Empowering Local Communities**:
**Issue**: Local communities often face challenges in effectively conveying their needs.
**Solution**: The model empowers individuals to communicate urgent requirements, ensuring that local voices are heard and addressed promptly.




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

