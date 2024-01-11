import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data_from_db(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    
    df = df.drop(['child_alone'],axis=1)
    
    df['related'] = df['related'].replace({2: 1})
    
    X = df['message']
    y = df.iloc[:,4:]
    
    category_names = y.columns # This will be used for visualization purpose
    return X, y, category_names
    
    


def tokenize(text, url_placeholder="urlplaceholder"):
    """
    Tokenization function for processing text.
    
    Args:
        text (str): Input text to be tokenized.
        url_placeholder (str): Placeholder string for URLs.
        
    Returns:
        clean_tokens (list): List of tokens extracted from the text.
    """
    
    # Identify and replace all URLs with a placeholder string
    url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_pattern, text)
    
    # Replace URLs with the specified placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_placeholder)

    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Lemmatize to remove inflectional and derivationally related forms of words
    lemmatizer = nltk.WordNetLemmatizer()

    # Generate a list of clean tokens
    clean_tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens]
    return clean_tokens
    
    


def build_pipeline():
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_processing', Pipeline([
            ('tokenization', CountVectorizer(tokenizer=tokenize)),
            ('tfidf_conversion', TfidfTransformer())
        ])),
        ('starting_verb_extraction', StartingVerbExtractor())
    ])),
    ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    
    ])
    parameters = {
        'clf__estimator__learning_rate':[0.5, 1.0],
        'clf__estimator__n_estimators':[10,20]
    
    }
        
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=3) 

    return cv
  


def multioutput_fscore(labels_true, labels_pred, beta=1):
    """
    Custom MultiOutput F-score Metric
    
    This metric calculates a sort of geometric mean of the F-beta score, computed on each label.
    
    It's designed for compatibility with multi-label and multi-class problems.
    The metric incorporates some peculiarities, such as the geometric mean and exclusion of 
    trivial solutions (e.g., 100% removal). This deliberate underestimation aims to handle 
    issues in multi-class/multi-label imbalanced cases.
    
    This function can be used as a scorer for GridSearchCV:
        scorer = make_scorer(custom_multioutput_fscore, beta=1)
        
    Parameters:
        labels_true (array-like): True labels.
        labels_pred (array-like): Predicted labels.
        beta (float): Beta value used to calculate the F-beta score.

    Returns:
        f1_score (float): Geometric mean of the F-beta scores.
    """
    
    # If provided predicted labels are in a DataFrame, extract the values
    if isinstance(labels_pred, pd.DataFrame):
        labels_pred = labels_pred.values
    
    # If provided true labels are in a DataFrame, extract the values
    if isinstance(labels_true, pd.DataFrame):
        labels_true = labels_true.values
    
    fbeta_scores = []
    for column in range(labels_true.shape[1]):
        score = fbeta_score(labels_true[:, column], labels_pred[:, column], beta, average='weighted')
        fbeta_scores.append(score)
        
    fbeta_scores = np.asarray(fbeta_scores)
    fbeta_scores = fbeta_scores[fbeta_scores < 1]
    
    # Calculate the geometric mean of F-beta scores
    f1_score = gmean(fbeta_scores)
    
    return f1_score

def evaluate_pipeline(pipeline, X_test, Y_test, category_names):
    """
    Evaluate the performance of a machine learning pipeline.

    Parameters:
        pipeline: The trained machine learning pipeline.
        X_test (array-like): Test features.
        Y_test (array-like): True labels for the test set.
        category_names (list): List of category names.

    Prints:
        - Average overall accuracy
        - F1 score using a custom definition
        - Classification report for each category
    """
    # Predictions using the provided pipeline
    Y_pred = pipeline.predict(X_test)
    
    # Calculate multioutput F1 score and overall accuracy
    multi_f1 = multioutput_fscore(Y_test, Y_pred, beta=1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    # Print evaluation metrics
    print('Average overall accuracy: {0:.2f}%'.format(overall_accuracy * 100))
    print('F1 score (custom definition): {0:.2f}%'.format(multi_f1 * 100))

    # Convert predictions to a DataFrame with appropriate column names
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns)
    
    # Display classification report for each category
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column], Y_pred[column]))
    


def save_model_as_pickle(pipeline, model_filepath):
    """
    Save Model to Pickle Function
    
    This function saves a trained model as a Pickle file, allowing for easy loading later.
    
    Parameters:
        pipeline: Trained model (e.g., GridSearchCV or Scikit-learn Pipeline object).
        model_filepath (str): Destination path to save the .pkl file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(pipeline, file)



def main():
    """
    Train Classifier Main Function
    
    This function applies the machine learning pipeline:
        1) Extracts data from SQLite database
        2) Trains the machine learning model on the training set
        3) Estimates model performance on the test set
        4) Saves the trained model as a Pickle file
        
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data_from_db(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        pipeline = build_pipeline()
        
        print('Training model...')
        pipeline.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_pipeline(pipeline, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model_as_pickle(pipeline, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
