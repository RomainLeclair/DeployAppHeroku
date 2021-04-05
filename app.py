import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

data = pd.read_csv('Sentiment.csv')
tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(data['text'].values)

class inputToModel(BaseModel):
    text:str


def preProcess_data(text):
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

def my_pipeline(text):
    text_new = preProcess_data(text)
    X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
    X = pad_sequences(X, maxlen=28)
    return X




app = FastAPI()

@app.get('/')
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict"}
@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''
        <form method="post">
        <input maxlength="28" name="text" type="text" value="Text Emotion to be tested" />
        <input type="submit" />'''

@app.post('/predict')
def predict(text:str = Form(...)):
    # clean and process text
    clean_text = my_pipeline(text)

    # Load and save model
    loaded_model = tf.keras.models.load_model('sentiment_1.h5')

    # predict text
    predictions = loaded_model.predict(clean_text)

    # Calculate the index of max sentiment
    sentiment = int(np.argmax(predictions))

    # Calculate the probability
    probability = max(predictions.tolist()[0])

    if sentiment==0:
        t_sentiment = 'negative'
    elif sentiment==1:
        t_sentiment = 'neutral'
    elif sentiment==2:
        t_sentiment = 'positive'
    return{
        "ACTUAL SENTENCES": text,
        "PREDICTED SENTIMENT": t_sentiment,
        "PROBABILITY": probability
    }


    