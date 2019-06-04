import sys
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import sqlalchemy as db
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    This function loads data from sqlite db into pandas dataframe by taking the db name as parameter
    
    parameters
    -----------
    database_filepath - sqllite db name
    
    return value
    -----------
    returns three parameters messages, catogory values, categories
    """
    database_filepath='sqlite:///'+database_filepath
    engine = db.create_engine(database_filepath)
    df = pd.read_sql_table("DisasterData",con=engine)
    x = df['message'].values
    y = df.iloc[:, 4:].values
    category_columns = (df.columns[4:]).tolist()
    return x,y,category_columns


def tokenize(text):
    """
    This function extracts the words and excludes the stop words
    
    parameters
    -----------
    text - text to be tokenized
    
    return value
    -----------
    returns list of clean_tokens
    """
    # applying regular experssion
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    #removing stop words
    tokens = [p for p in tokens if p not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    


def build_model():
    """
    This function builds the model and returns the same
    parameters
    -----------
    none
    return value
    -----------
    returns the model created
    """
    # building the model using random classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=1)),
    ])

    # grid serach parameters
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': [True, False],
        'tfidf__norm': ['l1', 'l2']
    }

    # optimizing the model;
    optimized_model = GridSearchCV(pipeline, param_grid=parameters,
                         cv=2, verbose=1)
    return optimized_model


def evaluate_model(model, x_test, y_test, category_names):
    """
    This function evaluates the models and prints the classification report
    parameters
    ----------------
    model - model to be evaluated
    x_test,y_test - test data to be predicted for
    category_names - catagories
    
    return value
    --------------------
    none
    """
    y_pred = model.predict(x_test)
    print("evaluating the model")
    for counter in range(len(category_names)):
        print("Label:", category_names[counter])
        print(classification_report(y_test[:, counter], y_pred[:, counter]))


def save_model(model, model_filepath):
    """
    This function saves the model to pkl file
    parameters
    ----------------
    model - model to be saved
    model_filepath - name of pkl file
    
    return value
    --------------------
    none
    """
    with open(model_filepath, 'wb') as outfile:
       pickle.dump(model, outfile)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()