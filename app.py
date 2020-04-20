
from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import re
import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle


movies_df = pd.read_csv('movies1.csv')
#movies_df=movies_df.loc[:,{'Title','Genre', 'Plot'}]
movies_df = movies_df.dropna()
with open('simDist.pkl','rb') as f: similarity_distance = pickle.load(f)


app = Flask(__name__)
#tlitle = request.form['message']
@app.route('/')
def home():
	return render_template('index1.html')

@app.route('/predict',methods=['POST'])

def predict():

    if request.method == 'POST':
        title = request.form['message']

        print(title in movies_df.Title)
        if title in movies_df.Title.tolist():

            #print(movies_df['Title'] == title)

            index = movies_df[movies_df['Title'] == title].index[0]
            vector = similarity_distance[index, :]
            most_similar = movies_df.iloc[np.argsort(vector)[1], 3]
            pred_text = most_similar + " is similar to "
            
 
        else:
            pred_text = "Please try a different movie!"
            title= " "

            

    return render_template('index1.html',prediction_text = pred_text ,message_text= title)



if __name__ == '__main__':
	app.run(debug=True)