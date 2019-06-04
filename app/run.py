import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import plotly.graph_objs as goj

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data from data folder
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterData', engine)
print("dataset created")
# load model pickle file from model file
model = joblib.load("../models/classifier.pkl")


# index webpage displays  visuals and receives user input text for model 
def category_graph():
    """this function creates bar graph for top categories
    parameters
    ----------------
    none
    returns value
    ----------------
    figure object
    """

    # summing on categories
    categories_display = df.iloc[:, 4:].sum().sort_values(ascending=False)
    categories_display=categories_display.head(20)

    data = [goj.Bar(
        x=categories_display.index,
        y=categories_display,
        marker=dict(color='gold'),
        opacity=0.8
    )]

    layout = goj.Layout(
        title="Counts for top 20 Categories ",
        xaxis=dict(
            title='categories',
            tickangle=90
        ),
        yaxis=dict(
            title='counts',
            tickfont=dict(
                color='black'
            )
        )
    )

    return goj.Figure(data=data, layout=layout)


def genre_graph():
    """
    This function creates genre stack graph for top 10 categories
    
    parameters
    -----------------------
    none
    
    return value
    *****************************
    figure object
    """

    # getting the data for genre graph
    categories = df.iloc[:, 4:].sum().sort_values(ascending=False)
    genres = df.groupby('genre').sum()[categories.index[:10]]
    
    color_bar = 'green'

    data = []
    for cat in genres.columns[1:]:
        data.append(goj.Bar(
                    x=genres.index,
                    y=genres[cat],
                    name=cat)
                    )

    layout = goj.Layout(
        title="Genre wise bar graph for top 10 categories",
        xaxis=dict(
            title='Distribution by Genre',
            tickangle=0
        ),
        yaxis=dict(
            title='category wise counts',
            tickfont=dict(
                color=color_bar
            )
        ),
        barmode='bar'
    )

    return goj.Figure(data=data, layout=layout)


# get figures and top categories
fig1 = category_graph()
fig2 = genre_graph()



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # encode plotly graphs in JSON
    graphs = [fig1, fig2]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()