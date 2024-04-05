#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append( '../util' )
import util as util
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset


# In[2]:


# creates and checks metadata object

def get_metadata(train):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train)
    metadata_dict = metadata.to_dict()
    print()
    print("METADATA:")
    for key, value in metadata_dict.items():
        print(key, value)
    print()
    print("COLUMNS:")
    columns = metadata_dict.get('columns')
    for key, value in columns.items():
        sdtype = value.get('sdtype')
        print(key + ": " + sdtype)
    
    return metadata


# In[5]:


def get_synthesizer():
    # train/test split, get metadata for train
    train, test = util.import_dataset(subset_frac=.02)
    metadata = get_metadata(train)


    # make and fit fast synthesizer
    fast_synthesizer = SingleTablePreset(metadata, name='FAST_ML')
    fast_synthesizer.fit(train) 
    filepath = "synthesizer_pickles/FML_synthesizer.pkl"
    fast_synthesizer.save(filepath=filepath)
    return fast_synthesizer

