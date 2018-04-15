#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 02:48:28 2018

@author: dhaval
"""

# Normal Imports
import pandas as pd
import numpy as np
import pickle
from processTweet import clean

# Sklearn Imports
from sklearn.feature_extraction.text import CountVectorizer

model = pickle.load(open('randomForest.model', 'rb'))
    
cv = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
x = pd.DataFrame(columns = ['Data'])
x.loc[0] = input('Enter Tweet: ')
x['Data'] = x['Data'].apply(lambda x : clean(x))
print(x['Data'])
X = x['Data'].as_matrix()
X = cv.transform(X).toarray()

modelPredictions = model.predict(X)