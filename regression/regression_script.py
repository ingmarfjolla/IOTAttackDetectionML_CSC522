# %%
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
from datetime import datetime
from joblib import effective_n_jobs

# %%

##profiling doesn't work here so commenting out
# @profile
def train_test_logistic_regression(train,test):
    y_test = test[util.y_column].values
    model = LogisticRegression(n_jobs=-1)
    print("Using n_jobs = ", effective_n_jobs(-1))
    print("Training regression model......")
    model.fit(train[util.X_columns],train[util.y_column])
    print("Testing regression model......")
    preds = model.predict(test[util.X_columns])
    print("Returning model predictions......")
    return preds,y_test



# %% [markdown]
# # Regression with 2 classes

# %%
start_time = datetime.now()
print("Starting 2 class model at: ", start_time.strftime("%H:%M"))
train,test = util.import_dataset(numclasses=2,use_subset=True,subset_frac=0.5)
y_pred, y_test = train_test_logistic_regression(train,test)
print()
print()
print(f"##### Regression (2 classes) #####")
print('accuracy_score: ', accuracy_score(y_pred, y_test))
print('recall_score: ', recall_score(y_pred, y_test, average='macro'))
print('precision_score: ', precision_score(y_pred, y_test, average='macro'))
print('f1_score: ', f1_score(y_pred, y_test, average='macro'))
print()
elapsed_time = datetime.now() - start_time
print("Elapsed time: ", elapsed_time)
print()
print()
print()
del train,test,y_pred,y_test


# # %% [markdown]
# # # Regression with 7 classes

# # %%
# start_time = datetime.now()
# print("Starting 7 class model at: ", start_time.strftime("%H:%M"))
# train,test = util.import_dataset(numclasses=7)
# y_pred, y_test = train_test_logistic_regression(train,test)
# print()
# print()
# print(f"##### Regression (7 classes) #####")
# print('accuracy_score: ', accuracy_score(y_pred, y_test))
# print('recall_score: ', recall_score(y_pred, y_test, average='macro'))
# print('precision_score: ', precision_score(y_pred, y_test, average='macro'))
# print('f1_score: ', f1_score(y_pred, y_test, average='macro'))
# print()
# elapsed_time = datetime.now() - start_time
# print("Elapsed time: ", elapsed_time)
# print()
# print()
# print()
# del train,test,y_pred,y_test

# # %% [markdown]
# # # Regression with 34 classes

# # %%


# # %%
# start_time = datetime.now()
# print("Starting 34 class model at: ", start_time.strftime("%H:%M"))
# train,test = util.import_dataset()
# y_pred, y_test = train_test_logistic_regression(train,test)
# print()
# print()
# print(f"##### Regression (34 classes) #####")
# print('accuracy_score: ', accuracy_score(y_pred, y_test))
# print('recall_score: ', recall_score(y_pred, y_test, average='macro'))
# print('precision_score: ', precision_score(y_pred, y_test, average='macro'))
# print('f1_score: ', f1_score(y_pred, y_test, average='macro'))
# print()
# elapsed_time = datetime.now() - start_time
# print("Elapsed time: ", elapsed_time)
# print()
# print()
# print()
# del train,test,y_pred,y_test


