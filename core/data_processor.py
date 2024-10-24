import math
import numpy as np
import pandas as pd

class DataLoader():

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        data_split = int(len(dataframe) * split) # Split data by 85% (split = 0.85 in config.json)

        self.data_train = dataframe.get(cols).values[:data_split] # Train first 85% of the data
        self.data_test = dataframe.get(cols).values[data_split:] # Test last 15% of the data

        self.len_train = len(self.data_train) # Length of the training data
        self.len_test = len(self.data_test) # Length of the test data

        self.len_train_windows = None