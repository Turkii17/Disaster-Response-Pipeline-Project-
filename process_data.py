import sys
import pandas as pd
from sqlalchemy import create_engine

def load_messages_with_categories(messages_filepath, categories_filepath):
    """
    Load data from CSV files and merge them on the 'id' column.

    Parameters:
    - messages_filepath (str): Filepath for the messages CSV file.
    - categories_filepath (str): Filepath for the categories CSV file.

    Returns:
    - df (DataFrame): Merged DataFrame containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_categories_data(df):
    """
    Clean the merged DataFrame by splitting categories, renaming columns, and converting data types.

    Parameters:
    - df (DataFrame): Merged DataFrame containing messages and categories.

    Returns:
    - df (DataFrame): Cleaned DataFrame.
    """
    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # Extract category names
    row = categories.iloc[[1]]
    category_colnames = list(map(lambda category_name: category_name.split('-')[0], row.values[0]))
    categories.columns = category_colnames

    # Convert category values to binary
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]
        categories[column] = categories[column].astype(int)

    # Drop the original 'categories' column and concatenate the new categories
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], join='inner', axis=1)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data_to_db(df, database_filename):
    """
    Save the cleaned DataFrame to a SQLite database.

    Parameters:
    - df (DataFrame): Cleaned DataFrame.
    - database_filename (str): Filepath for the SQLite database.
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine,if_exists = 'replace', index=False)

def main():
    """
    Main function that orchestrates the ETL process.

    It loads, cleans, and saves data based on user-provided filepaths.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_messages_with_categories(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_categories_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data_to_db(df, database_filepath)

        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
