from ctgan import CTGAN
import pandas as pd
import sys
sys.path.append( '../util' )
import util as util
from sklearn.preprocessing import LabelEncoder
# import pdb; 
labelencoder = LabelEncoder()

discrete_columns = [util.y_column, 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 
                    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']

train, test = util.import_dataset(7,"dnn",True,.1)
train[discrete_columns] = labelencoder.fit_transform(discrete_columns)


ctgan = CTGAN(epochs=1,verbose=True)
print("Starting to train models...")
# pdb.set_trace()
ctgan.fit(train, discrete_columns)
# Create synthetic data
synthetic_data = ctgan.sample(10)