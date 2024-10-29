import os
import datetime as dt

from numpy import newaxis
from core.utils import Timer
from keras.layers import Dense, Dropout, LSTM  
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Model():
    """ A class for building and inferencing an lstm model """

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        
        print('[Model] Loading model from file %s' % filepath)

        self.model = load_model(filepath)

        # This function loads a previously saved Keras model from filepath. 
		# Useful for continuing training or making predictions using a pre-trained model.

    def build_model(self, config):

        timer = Timer()
        timer.start

        for layer in config['model']['layers']:

            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation = activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape = (input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
        
        self.model.compile(loss = config['model']['loss'], optimizer = config['model']['optimizer'])

        print('[Model] Training Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):

        timer = Timer()
        timer.start()

        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        save_name = os.path.join(save_dir, '%s-e%s.keras' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))

        callbacks = [
            EarlyStopping(monitor = 'val_loss', patience = 2),
			ModelCheckpoint(filepath = save_name, monitor = 'val_loss', save_best_only = True)
        ]

        self.model.fit(
            x, 
            y, 
            epochs = epochs,
            batch_size = batch_size,
            callbacks = callbacks
        )

        self.model.save(save_name)

        print('[Model] Training Completed. Model saved as %s' % save_name)
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):

        timer = Timer()
        timer.start()

        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

        save_name = os.path.join(save_dir, '%s-e%s.keras' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))

        callbacks = [
            ModelCheckpoint(filepath = save_name, monitor = 'loss', save_best_only = True)
        ]

        self.model.fit_generator(
            data_gen,
            steps_per_epoch = steps_per_epoch,
			epochs = epochs,
			callbacks = callbacks,
			workers = 1
        )

        print('[Model] Training Completed. Model saved as %s' % save_name)
        timer.stop()

    def predict_sequence_live(self, data, window_size):

        print('[Model] Predicting Sequences live...')

        curr_frame = data[0] # Iindex to first window of incoming data
        predicted = []
        predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])

        return predicted