#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fast_ML_synthesizer
import sys
sys.path.append( '../regression')
import regression as regression


# In[ ]:


synthesizer, test = fast_ML_synthesizer.get_synthesizer()
baseline_set = synthesizer.sample(num_rows=38000000, batch_size=1000, output_file_path='synthetic_datasets/FML_baseline_set.csv')

y_pred, y_test = regression.train_test_logistic_regression(baseline_set, test)
print('### SYNTHETIC ###')
print()
regression.print_scores(y_pred, y_test)





