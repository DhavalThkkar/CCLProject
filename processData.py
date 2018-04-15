#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 00:22:22 2017

@author: dhaval
"""

# Importing the Libraries
import pandas as pd
import numpy as np
import sys
from processTweet import clean
import pickle

# Sklearn Imports
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer

# Printing on screen while kernel is running
def to_screen(msg):
    sys.stdout.flush
    sys.stdout.write(msg)
    
# Changing the Postive sentiment value '4' to '1'
def change(x):
    if x == 4:
        return 1
    else:
        return 0

def load_data(samples = 5000):
    # Importing data
    to_screen('Importing Data ...\n')
    col_name = ['Sentiment', 'ID', 'Date', 'NA', 'Author', 'Tweet']
    data = pd.read_csv('training.1600000.processed.noemoticon.csv', names=col_name, encoding='ISO-8859-1')
    data = data[['Sentiment', 'Tweet']]
    
    data['Sentiment'] = data['Sentiment'].apply(lambda x : change(x))
    
    # Separating into positive and negative samples for shuffling
    pos_samples = data[data['Sentiment'] == 1]
    neg_samples = data[data['Sentiment'] == 0]
    pos_samples = pos_samples.iloc[np.random.choice(len(pos_samples), size=samples)]
    neg_samples = neg_samples.iloc[np.random.choice(len(neg_samples), size=samples)]
    
    # Data variable takes up a lot of space so deleting
    del data
    
    # Concatenating Pos and Neg data into one dataframe as  million tweets can't be used for training
    data = pd.concat([pos_samples,neg_samples], axis = 0)
    data = data.reset_index()
    data = data[['Sentiment', 'Tweet']]
    del pos_samples, neg_samples
    
    data['CleanedTweets'] = data['Tweet'].apply(lambda x : clean(x))
    data = shuffle(data)
    
    # Dividing the set in X and y
    X = data['CleanedTweets'].as_matrix()
    y = data['Sentiment'].as_matrix()
    
    cv = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
    X = cv.fit_transform(X).toarray()
    
    return X, y

def loadTweetData(data):
    
    data = data['CleanedTweets'].as_matrix()
    cv = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
    data = cv.fit_transform(data).toarray()
    
    return data
    