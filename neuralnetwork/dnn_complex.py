#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


def create_multiclass_classification_model(input_shape, num_classes):
    inputs = Input(shape=(input_shape,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x) 
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[3]:


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
print("Starting 7 class model at: ", start_time.strftime("%H:%M"))
model = create_multiclass_classification_model(len(util.X_columns),num_classes)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#https://keras.io/api/models/model_training_apis/
history = model.fit(x=X_train, y=y_train_encoded,
                    validation_split=0.2, epochs=100, 
                    batch_size=256, callbacks=[early_stopping])


test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=2)
elapsed_time = datetime.now() - start_time
print("Elapsed time: ", elapsed_time)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
del X_train,X_test,y_train_encoded,y_test_encoded


# In[4]:


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


# In[8]:


###Change this name so you don't overrwrite the one we have now
model.save('./dnn_model.keras')


# In[10]:


from tensorflow.keras.models import load_model
loaded_model = load_model('./dnn_model.keras')
print(loaded_model.summary())
print(loaded_model.get_config())


# In[14]:


for layer in loaded_model.layers:
    weights = layer.get_weights()  
    #ValueError: not enough values to unpack (expected 2, got 0) <- fixing this error, not all llayers have bias or weight
    if len(weights) > 0:
        print(f"{layer.name} weights shape: {weights[0].shape}")
        if len(weights) > 1:
            print(f"{layer.name} biases shape: {weights[1].shape}")
    else:
        print(f"{layer.name} has no weights or biases.")


# In[22]:


from tensorflow.keras.utils import plot_model



# In[23]:


plot_model(loaded_model, to_file='./modelpics/model_plot.png', show_shapes=True, show_layer_names=True)

