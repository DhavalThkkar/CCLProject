#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:57:48 2018

@author: dhaval
"""


import pandas as pd
import numpy as np
import pickle
from processTweet import clean

# Sklearn Imports
from sklearn.feature_extraction.text import CountVectorizer

# Flask Imports
from flask import Flask, jsonify

app = Flask(__name__)
model = pickle.load(open('randomForest.model', 'rb'))
cv = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))

@app.route('/query=<tweet>')
def detail(tweet):
    x = pd.DataFrame(columns = ['Data'])
    x.loc[0] = tweet
    x['Data'] = x['Data'].apply(lambda x : clean(x))
    print(x['Data'])
    X = x['Data'].as_matrix()
    X = cv.transform(X).toarray()
    modelPredictions = model.predict(X)
    return jsonify({'Prediction': modelPredictions.tolist()})

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True, port = 8888)