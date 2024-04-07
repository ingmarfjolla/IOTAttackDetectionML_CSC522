from ctgan import CTGAN
import pandas as pd
import sys
sys.path.append( '../util' )
import util as util
from sklearn.preprocessing import LabelEncoder


discrete_columns = [util.y_column, 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 
                    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']

train, test = util.import_dataset(7,"dnn",True,.02)

for column in discrete_columns:
    le = LabelEncoder()
    train[column] = le.fit_transform(train[column])
    print(train[column].unique())

ctgan = CTGAN(epochs=1,verbose=True)
print("Starting to train models...")
# pdb.set_trace()
ctgan.fit(train, discrete_columns)
# Create synthetic data
synthetic_data = ctgan.sample(10)