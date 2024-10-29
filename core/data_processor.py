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

    def get_test_data(self, seq_len, normalise):

        data_windows = []

        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len]) # Prepare data into sequence of length 8

        data_windows = np.array(data_windows).astype(float) # Convert the data into a numpy array
        data_windows = self.normalise_windows(data_windows, single_window = False) if normalise else data_windows

        x = data_windows[:, :, -1] # Get 3D array
        y = data_windows[:, -1, [0]] # Get 2D array

        return x, y
    
       # Windowing: The data is split into windows (sequences) of a fixed length (seq_len). 
       # Each window will act as input for the LSTM.

    def get_train_data(self, seq_len, normalise):

        data_x = []
        data_y = []

        for i in range (self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise) # Creates windowed data of length 8 (seq_length)
            data_x.append(x) # Append input sequence
            data_y.append(y) # Append output sequence

        return np.array(data_x), np.array(data_y) # Returns the results as numpy array
    
        # Example:
        
        #First window: Input (x): [1, 2, 3, 4, 5, 6, 7], Target (y): [8]
        #Second window: Input (x): [2, 3, 4, 5, 6, 7, 8], Target (y): [9]
        #Third window: Input (x): [3, 4, 5, 6, 7, 8, 9], Target (y): [10]

    def generate_train_batch(self, seq_len, batch_size, normalise):
        
        # Yield a generator of training data from filename on given list of cols split for train/test

        i = 0

        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []

            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # Stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)
    
    def _next_window(self, i, seq_len, normalise):
        # Generates the next data window from the given index location i

        window = self.data_train[i:i+seq_len] # Extracts a window of `seq_len` from the training data starting at index `i`
        window = self.normalise_windows(window, single_window = True)[0] if normalise else window
        x = window[:-1] # Inputs: all but the last value
        y = window[-1, [0]] # Target: last value in the window

        return x, y
    
    def normalise_windows(self, window_data, single_window = False):

        normalised_data = []
        
        window_data = [window_data] if single_window else window_data

        for window in window_data:
            normalised_window = []

            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]

                # Normalized value = (current value / first value ) - 1

                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # Reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)

        return np.array(normalised_data)