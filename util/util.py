import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from memory_profiler import profile
import gc

DATASET_DIRECTORY = '../csvfiles/'

X_columns = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
       'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
       'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
       'ece_flag_number', 'cwr_flag_number', 'ack_count',
       'syn_count', 'fin_count', 'urg_count', 'rst_count', 
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
       'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
       'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
       'Radius', 'Covariance', 'Variance', 'Weight', 
]
y_column = 'label'


dict_2classes = {}
dict_2classes['DDoS-RSTFINFlood'] = 'Attack'
dict_2classes['DDoS-PSHACK_Flood'] = 'Attack'
dict_2classes['DDoS-SYN_Flood'] = 'Attack'
dict_2classes['DDoS-UDP_Flood'] = 'Attack'
dict_2classes['DDoS-TCP_Flood'] = 'Attack'
dict_2classes['DDoS-ICMP_Flood'] = 'Attack'
dict_2classes['DDoS-SynonymousIP_Flood'] = 'Attack'
dict_2classes['DDoS-ACK_Fragmentation'] = 'Attack'
dict_2classes['DDoS-UDP_Fragmentation'] = 'Attack'
dict_2classes['DDoS-ICMP_Fragmentation'] = 'Attack'
dict_2classes['DDoS-SlowLoris'] = 'Attack'
dict_2classes['DDoS-HTTP_Flood'] = 'Attack'

dict_2classes['DoS-UDP_Flood'] = 'Attack'
dict_2classes['DoS-SYN_Flood'] = 'Attack'
dict_2classes['DoS-TCP_Flood'] = 'Attack'
dict_2classes['DoS-HTTP_Flood'] = 'Attack'


dict_2classes['Mirai-greeth_flood'] = 'Attack'
dict_2classes['Mirai-greip_flood'] = 'Attack'
dict_2classes['Mirai-udpplain'] = 'Attack'

dict_2classes['Recon-PingSweep'] = 'Attack'
dict_2classes['Recon-OSScan'] = 'Attack'
dict_2classes['Recon-PortScan'] = 'Attack'
dict_2classes['VulnerabilityScan'] = 'Attack'
dict_2classes['Recon-HostDiscovery'] = 'Attack'

dict_2classes['DNS_Spoofing'] = 'Attack'
dict_2classes['MITM-ArpSpoofing'] = 'Attack'

dict_2classes['BenignTraffic'] = 'Benign'

dict_2classes['BrowserHijacking'] = 'Attack'
dict_2classes['Backdoor_Malware'] = 'Attack'
dict_2classes['XSS'] = 'Attack'
dict_2classes['Uploading_Attack'] = 'Attack'
dict_2classes['SqlInjection'] = 'Attack'
dict_2classes['CommandInjection'] = 'Attack'

dict_2classes['DictionaryBruteForce'] = 'Attack'


dict_7classes = {}
dict_7classes['DDoS-RSTFINFlood'] = 'DDoS'
dict_7classes['DDoS-PSHACK_Flood'] = 'DDoS'
dict_7classes['DDoS-SYN_Flood'] = 'DDoS'
dict_7classes['DDoS-UDP_Flood'] = 'DDoS'
dict_7classes['DDoS-TCP_Flood'] = 'DDoS'
dict_7classes['DDoS-ICMP_Flood'] = 'DDoS'
dict_7classes['DDoS-SynonymousIP_Flood'] = 'DDoS'
dict_7classes['DDoS-ACK_Fragmentation'] = 'DDoS'
dict_7classes['DDoS-UDP_Fragmentation'] = 'DDoS'
dict_7classes['DDoS-ICMP_Fragmentation'] = 'DDoS'
dict_7classes['DDoS-SlowLoris'] = 'DDoS'
dict_7classes['DDoS-HTTP_Flood'] = 'DDoS'

dict_7classes['DoS-UDP_Flood'] = 'DoS'
dict_7classes['DoS-SYN_Flood'] = 'DoS'
dict_7classes['DoS-TCP_Flood'] = 'DoS'
dict_7classes['DoS-HTTP_Flood'] = 'DoS'


dict_7classes['Mirai-greeth_flood'] = 'Mirai'
dict_7classes['Mirai-greip_flood'] = 'Mirai'
dict_7classes['Mirai-udpplain'] = 'Mirai'

dict_7classes['Recon-PingSweep'] = 'Recon'
dict_7classes['Recon-OSScan'] = 'Recon'
dict_7classes['Recon-PortScan'] = 'Recon'
dict_7classes['VulnerabilityScan'] = 'Recon'
dict_7classes['Recon-HostDiscovery'] = 'Recon'

dict_7classes['DNS_Spoofing'] = 'Spoofing'
dict_7classes['MITM-ArpSpoofing'] = 'Spoofing'

dict_7classes['BenignTraffic'] = 'Benign'

dict_7classes['BrowserHijacking'] = 'Web'
dict_7classes['Backdoor_Malware'] = 'Web'
dict_7classes['XSS'] = 'Web'
dict_7classes['Uploading_Attack'] = 'Web'
dict_7classes['SqlInjection'] = 'Web'
dict_7classes['CommandInjection'] = 'Web'


dict_7classes['DictionaryBruteForce'] = 'BruteForce'


# @profile
def encode_column(encode_column, num_classes, class_type):
    
    valid_classes = {2, 7}

    if num_classes not in valid_classes:
        return ValueError(f"Result: num_classes must be one of {valid_classes}")
    
    if num_classes == 2:
        encode_column = encode_column.map(dict_2classes).astype(class_type)

    if num_classes == 7:
        encode_column = encode_column.map(dict_7classes).astype(class_type)

    return encode_column


# @profile
def import_dataset(numclasses=34,modeltype="regression",subset_frac=1.0):
    print("Importing datasets.....")
    df_sets_generator = (k for k in os.listdir(DATASET_DIRECTORY) if k.endswith('.csv'))
    subset_size = int(len(os.listdir(DATASET_DIRECTORY)) * subset_frac)
    df_sets = []
    for _ in range(subset_size):
        try:
            df_sets.append(next(df_sets_generator))
        except StopIteration:
            break
    df_sets.sort()
    training_sets = df_sets[:int(len(df_sets)*.8)]
    test_sets = df_sets[int(len(df_sets)*.8):]
    print("Processing training and test data....")
    train = read_dataset(training_sets,modeltype)
    test = read_dataset(test_sets,modeltype)
    # print(train.dtypes)
    print("Finished processing training and test data")
    if numclasses == 2:
        print("Mapping labels to two classes.....")
        # train[y_column] = train[y_column].map(dict_2classes).astype('category')
        train[y_column] = encode_column(encode_column = train[y_column], num_classes = 2, class_type = 'category')
        # print(train[y_column].cat.categories)
        # test[y_column] = test[y_column].map(dict_2classes).astype('category')
        test[y_column] = encode_column(encode_column = test[y_column], num_classes = 2, class_type = 'category')

    if numclasses == 7:
        print("Mapping labels to seven classes.....")
        # train[y_column] = train[y_column].map(dict_7classes).astype('category')
        train[y_column] = encode_column(encode_column = train[y_column], num_classes = 7, class_type = 'category')
        # print(train[y_column].cat.categories)
        # test[y_column] = test[y_column].map(dict_7classes).astype('category')
        test[y_column] = encode_column(encode_column = test[y_column], num_classes = 7, class_type = 'category')
    print("Data import and processing complete....")
    return train,test

# @profile
def read_dataset(dataset,modeltype):

    ###to be represented as booleans###
    protocol_cols = ['HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 
                    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']
    ###to be represented as booleans### 

    scaler = StandardScaler()
    dataframes = []
    count = 0
    for file in dataset:
        df = pd.read_csv(DATASET_DIRECTORY + file)
        #if it's a regression, i'd like to keep these as boolean. DNN's don't take bool so left it as float
        # if(modeltype=="regression"):
        #     df[protocol_cols] = df[protocol_cols].astype(bool)
        float_cols = df.select_dtypes(include=['float64']).columns.difference(protocol_cols)
        #float32 makes it use way less memory
        df[float_cols] = df[float_cols].astype('float32')
        dataframes.append(df)
        count += 1
        
    

    combined_df = pd.concat(dataframes, ignore_index=True)
    del dataframes,float_cols
    gc.collect()


    
    combined_df[y_column] = combined_df[y_column].astype('category')
    #############VALIDATING PROCESSING DEBUGGING ####################
    # print(combined_df.dtypes)
    # print(combined_df[y_column].dtype)
    # print(combined_df[y_column].cat.categories)
    #############VALIDATING PROCESSING DEBUGGING ####################




    
    scaler_columns = combined_df.select_dtypes(include=['float32']).columns.difference([y_column])
    combined_df[scaler_columns] = scaler.fit_transform(combined_df[scaler_columns])
    #############VALIDATING PROCESSING DEBUGGING ####################
    # scaled_stats = combined_df[scaler_columns].describe()
    # print(scaled_stats)
    #############VALIDATING PROCESSING DEBUGGING ####################
    del scaler_columns
    gc.collect()
    #combined_df[X_columns] = scaler.fit_transform(combined_df[X_columns])
    return combined_df


    
