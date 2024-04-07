#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import synthesizer as synthesizer
import sys
sys.path.append( '../regression')
import regression as regression


# In[ ]:


synthesizer, test = synthesizer.get_synthesizer(stype="gcs", name="GC_synthesizer.pkl")
baseline_set = synthesizer.sample(num_rows=38000000, batch_size=1000, output_file_path='../synthetic_data/synthetic_datasets/GC_baseline_set.csv')

y_pred, y_test = regression.train_test_logistic_regression(baseline_set, test)
print('### GaussianCopula Synthesizer, Logistic Regression Classifer ###')
print()
regression.print_scores(y_pred, y_test)