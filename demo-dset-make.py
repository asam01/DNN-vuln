import os
import pandas as pd
import numpy as np
import torch
import pickle
from gluonts.time_feature import get_lags_for_frequency, time_features_from_frequency_str

def create_sequences(df, label, sequence_length, prediction_length, is_test):#, sequence_length=1, prediction_length=1):
    """
    For a given dataframe and sequence length, this function will create sequences
    for sequence-to-sequence prediction. It includes the raw time as a feature.
    Each feature within a sequence has the same timestamp. Each predicted future value
    also has an associated timestamp.
    """
    start_datetime = pd.Timestamp('2023-11-07')
    def convert_to_pandas_period(time, base_datetime, freq):
        # Convert the stopwatch time to a datetime by adding the time as a timedelta to the base datetime
        datetime = base_datetime + pd.Timedelta(seconds=time)
        # Convert the datetime to the period of the desired frequency
        return pd.Period(datetime, freq)

    def transform_time_field_to_period(df, base_datetime, freq):
        df["time"] = [convert_to_pandas_period(time, base_datetime, freq) for time in df["time"]]
        return df

    df['time'] = df['time'] * 10

    freq = "S"
    df = transform_time_field_to_period(df, start_datetime, freq)

    timestamps = list(df['time'])
    df = df[['time', 'branch-misses']] # univariate for initial testing
    targets = [row[df.columns.difference(['time'])].tolist() for _, row in df.iterrows()]
    targets = [item for sublist in targets for item in sublist] # flatten

    if is_test:
        return timestamps[0], targets[:sequence_length+prediction_length], label
    else:
        return timestamps[0], [targets[:sequence_length], targets[:sequence_length+prediction_length]], label

def preprocess_and_create_datasets(directory, eval_every_n=5):
    """
    This function will preprocess all CSV files in the given directory,
    separate them into training and evaluation datasets, and
    create Hugging Face datasets from them.
    """
    # Initialize lists to store sequences for training and evaluation
    train_sequences = []
    eval_sequences = []
    f_files_counter = 0  # Counter to track how many 'f' files have been processed
    
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            label = 'f' if filename.startswith('f') else 't'
            df = pd.read_csv(os.path.join(directory, filename))

            # Drop any columns where the name starts with 'Unnamed'
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # Drop specified columns
            columns_to_drop = ['LLC-loads', 'LLC-load-misses', 'L1-dcache-prefetch-misses']
            df = df.drop(columns=columns_to_drop)#, errors='ignore')

            # Drop rows with NaN values
            df = df.dropna()

            # Ensure data is in the correct order
            df = df.sort_values('time')

            if len(df) < 8:
                continue

            if label == 'f': # goes in training and eval set
                seq_train_eval = create_sequences(df, label, sequence_length=5, prediction_length=3, is_test=False)
                seq_train = (seq_train_eval[0], seq_train_eval[1][0], seq_train_eval[2])
                seq_eval = (seq_train_eval[0], seq_train_eval[1][1], seq_train_eval[2])
            else: # goes in test set
                seq_test = create_sequences(df, label, sequence_length=5, prediction_length=3, is_test=True)

            # # Arbitrarily choose 'f' files for eval set using a counter
            # fpath = 'dataset/eval/'
            # if label == 'f':
            #     if f_files_counter % eval_every_n == 0:  # Every nth 'f' file goes into the eval set
            #         eval_sequences.extend(seq)
            #     else:
            #         train_sequences.extend(seq)
            #         fpath = 'dataset/train/'
            #         sequence_length = 5 # use 5 timestamps to predict next values
            #     f_files_counter += 1
            # else:  # All 't' files go into the eval set
            #     eval_sequences.extend(seq)

            # Create sequences for the file
            # seq = create_sequences(df, label, sequence_length)
            # seq = (timestamp, targets) 
            # seq = (seq[0], seq[1], label)
            if label == 'f':
                with open('dataset/train/' + filename + '.pkl', 'wb') as file:
                        pickle.dump(seq_train, file) 
                with open('dataset/eval/' + filename + '.pkl', 'wb') as file:
                        pickle.dump(seq_eval, file) 
            else:
                with open('dataset/test/' + filename + '.pkl', 'wb') as file:
                        pickle.dump(seq_test, file)          

directory = "do_snoopy/"
preprocess_and_create_datasets(directory)
