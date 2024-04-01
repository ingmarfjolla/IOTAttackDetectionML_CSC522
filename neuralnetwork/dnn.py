#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import gc
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sys
sys.path.append( '../util' )
import util as util
from datetime import datetime


# # DNN with 2 classes

# In[2]:


##simplifying this 
# def create_binary_classification_model(input_shape):
#     inputs = Input(shape=(input_shape,))
#     x = Dense(128, activation='relu')(inputs)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     x = Dense(64, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     outputs = Dense(1, activation='sigmoid')(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer=Adam(learning_rate=0.001),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

def create_binary_classification_model(input_shape):
    inputs = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(inputs)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

train, test = util.import_dataset(2,"dnn")



y_train = train[util.y_column]
y_test = test[util.y_column]

X_train = train.drop(util.y_column, axis=1)
X_test = test.drop(util.y_column, axis=1)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

del train,test,y_train,y_test

start_time = datetime.now()
print("Starting model at: ", start_time.strftime("%H:%M"))
model = create_binary_classification_model(len(util.X_columns))
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#https://keras.io/api/models/model_training_apis/
history = model.fit(x=X_train, y=y_train_encoded,
                    validation_split=0.2, epochs=100, 
                    batch_size=256, callbacks=[early_stopping])


test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=2)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
elapsed_time = datetime.now() - start_time
print("Elapsed time: ", elapsed_time)
del X_train,X_test,y_train,y_test


# In[3]:


import matplotlib.pyplot as plt
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()


# # 7 Class classifier DNN

# In[12]:


def create_multiclass_classification_model(input_shape, num_classes):
    inputs = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(inputs)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[13]:


train, test = util.import_dataset(7,"dnn")
y_train = train[util.y_column]
y_test = test[util.y_column]

X_train = train.drop(util.y_column, axis=1)
X_test = test.drop(util.y_column, axis=1)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

del train,test,y_train,y_test
num_classes = len(label_encoder.classes_)
print(" Number of classes is:" )
print(num_classes)
print("")
start_time = datetime.now()
print("Starting model at: ", start_time.strftime("%H:%M"))
model = create_multiclass_classification_model(len(util.X_columns),num_classes)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#https://keras.io/api/models/model_training_apis/
history = model.fit(x=X_train, y=y_train_encoded,
                    validation_split=0.2, epochs=100, 
                    batch_size=256, callbacks=[early_stopping])


test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=2)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
elapsed_time = datetime.now() - start_time
print("Elapsed time: ", elapsed_time)
del X_train,X_test,y_train_encoded,y_test_encoded


# In[14]:


import matplotlib.pyplot as plt
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy for 7 classes')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()


# # 34 classes

# In[15]:


train, test = util.import_dataset(34,"dnn")
y_train = train[util.y_column]
y_test = test[util.y_column]

X_train = train.drop(util.y_column, axis=1)
X_test = test.drop(util.y_column, axis=1)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

del train,test,y_train,y_test
num_classes = len(label_encoder.classes_)
print(" Number of classes is:" )
print(num_classes)
print("")
start_time = datetime.now()
print("Starting model at: ", start_time.strftime("%H:%M"))
model = create_multiclass_classification_model(len(util.X_columns),num_classes)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#https://keras.io/api/models/model_training_apis/
history = model.fit(x=X_train, y=y_train_encoded,
                    validation_split=0.2, epochs=100, 
                    batch_size=256, callbacks=[early_stopping])


test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=2)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
elapsed_time = datetime.now() - start_time
print("Elapsed time: ", elapsed_time)
del X_train,X_test


# In[16]:


import matplotlib.pyplot as plt
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy for 34 classes')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

