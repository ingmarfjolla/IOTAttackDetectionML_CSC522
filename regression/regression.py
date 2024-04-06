#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
sys.path.append( '../util' )
import util as util
from memory_profiler import profile


# In[3]:


##profiling doesn't work here so commenting out
# @profile
def train_test_logistic_regression(train,test):
    y_test = test[util.y_column].values
    model = LogisticRegression(n_jobs=-1)
    print("Training regression model......")
    model.fit(train[util.X_columns],train[util.y_column])
    print("Testing regression model......")
    preds = model.predict(test[util.X_columns])
    print("Returning model predictions......")
    return preds,y_test


def print_scores(y_pred, y_test):
    print('accuracy_score: ', accuracy_score(y_pred, y_test))
    print('recall_score: ', recall_score(y_pred, y_test, average='macro'))
    print('precision_score: ', precision_score(y_pred, y_test, average='macro'))
    print('f1_score: ', f1_score(y_pred, y_test, average='macro'))
    print()
    print()
    print()   
    