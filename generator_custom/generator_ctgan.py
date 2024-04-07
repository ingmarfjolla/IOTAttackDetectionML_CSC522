from ctgan import CTGAN
import pandas as pd
import sys
sys.path.append( '../util' )
import util as util
from sklearn.preprocessing import LabelEncoder

def get_ctgan(subset_frac=1.0):
    discrete_columns = [util.y_column, 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 
                        'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']

    train, test = util.import_dataset(7,"dnn",subset_frac=subset_frac)

    del test

    for column in discrete_columns:
        le = LabelEncoder()
        train[column] = le.fit_transform(train[column])
        print(train[column].unique())

    ctgan = CTGAN(epochs=25,verbose=True)
    print("Starting to train models...")
    # pdb.set_trace()
    ctgan.fit(train, discrete_columns)


    fraction = str(subset_frac)

    ctgan.save(fraction + "_subset_ctgan.pkl")