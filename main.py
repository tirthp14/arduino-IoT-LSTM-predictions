import os
import json
import matplotlib.pyplot as plt
import serial
import numpy as np

from core.data_processor import DataLoader
from core.model import Model

fig = plt.figure(facecolor = 'white')

def plot_results(predicted_data, true_data):

    plt.cla() # Clear previous fig

    ax = fig.add_subplot(111)
    ax.plot(true_data, label = 'True Data')

    plt.plot(predicted_data, label = 'Prediction')
    plt.legend()

def plot_results_multiple(predicted_data, true_data, prediction_len):

    fig = plt.figure(facecolor = 'white')

    ax = fig.add_subplot(111)
    ax.plot(true_data, label = 'True Data')

    # Pad the list of predictions to shift it in the graph to it's correct start

    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]

        plt.plot(padding + data, label = 'Predicition')
        plt.legend()

    plt.show()

def main():

    config = json.load(open('config.json', 'r'))
    if not os.path.exists(config['model']['save_directory']): os.makedirs(config['model']['save_directory'])

    model = Model()
    model.build_model(config)

    # Get live sensor data from Arduino and predict the next 10 sensor data

    sensor_port = serial.Serial('COM3', 9600)
    sensor_port.close()
    sensor_port.open()
    seq_len = config['data']['sequence_length']
    sensor_data = []
    predictions_data = []
    live_data = np.zeros(seq_len - 1)


    plt.ion() # Real time graph

    while True:

        i = 0
        
        while i < seq_len - 1:                       # Store incoming data to testing data array
            b = sensor_port.readline()                  # Read a byte string
            live_data[i] = float(b.decode())
            sensor_data.append(live_data[i])
            i += 1

        sensor_struct_data = live_data[np.newaxis,:,np.newaxis] # Contruct live data for LSTM
        predictions= model.predict_sequence_live(sensor_struct_data, config['data']['sequence_length'])

        #Shift the window by 1 new prediction each time, re-run predictions on new window

        predictions_data.append(predictions)

        plot_results(predictions_data[-120:], sensor_data[-100:])

        plt.show()
        plt.pause(0.1)

        if len(sensor_data) > 10 * seq_len:
            np.savetxt('data\sensor.csv', sensor_data, delimiter = ',', header = 'sensor_value')
        
        # Load data for training

            data = DataLoader(
                os.path.join('data', config['data']['filename']),
                config['data']['tt_split'],
                config['data']['columns']
            )

            x, y = data.get_train_data(
                seq_len = config['data']['sequence_length'],
                normalise = config['data']['normalise']
            )

            model.train(
                x, 
                y, 
                epochs = config['training']['epochs'],
                batch_size = config['training']['batch_size'],
                save_dir = config['model']['save_directory']
            )

            sensor_data = sensor_data[-100:]

if __name__ == '__main__':
    main()