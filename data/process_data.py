import sys
import pandas as pd
import sqlalchemy as db
import os


def load_data(messages_filepath, categories_filepath):
    """
    This function loads data from two csv files passed as parameters and merge them
    on id column and return this dataset.
    parameters
    -----------
    messages_filepath - messages file path
    categories_filepath - categories file path
    return value
    -----------
    returns the dataset created
    """
    messages = pd.DataFrame(pd.read_csv(messages_filepath))
    categories = pd.DataFrame(pd.read_csv(categories_filepath))
    # merge datasets
    df = pd.merge(messages,categories, on='id')
    return df


def clean_data(df):
    """
    This function cleans the dataset  passed as parameters returns the same
    parameters
    -----------
    df -dataset to be cleaned
   
    return value
    -----------
    returns the cleaned dataset
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    #header creation for the categories dataset
    # select the first row of the categories dataframe
    row = categories.head(1)
    # use this row to extract a list of new column names for categories.
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
    categories.columns = category_colnames
    categories.head()
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        # drop the original categories column from `df`
    df=df.drop(['categories'],axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    #find the totol rows and columns
    print('rows=',df.shape[0],',columns=',df.shape[1])
    # drop duplicates
    df=df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    This function saves the dataset to a sqlite DB file passed as a parameter
    parameters
    -----------
    df - dataset to be saved to sqlite db
    database_filename - sqlite db name
    return value
    -----------
    none
    """
    db_name='sqlite:///'+database_filename
    print(db_name)
    os.remove(database_filename)
    engine = db.create_engine(db_name)
    df.to_sql('DisasterData', engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()