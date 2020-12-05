import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from tensorflow.keras.activations import tanh, elu, relu
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence

import os
from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import math
import h5py
import argparse

   
def pre_emphasis_filter(x, coeff=0.95):
    return tf.concat([x, x - coeff * x], 1)
    
def error_to_signal(y_true, y_pred): 
    """
    Error to signal ratio with pre-emphasis filter:
    """
    y_true, y_pred = pre_emphasis_filter(y_true), pre_emphasis_filter(y_pred)
    return K.sum(tf.pow(y_true - y_pred, 2), axis=0) / K.sum(tf.pow(y_true, 2), axis=0) + 1e-10
    
def save_wav(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.float32))

def normalize(data):
    data_max = max(data)
    data_min = min(data)
    data_norm = max(data_max,abs(data_min))
    return data / data_norm

def main(args):
    '''Ths is a similar Tensorflow/Keras implementation of the LSTM model from the paper:
        "Real-Time Guitar Amplifier Emulation with Deep Learning"
        https://www.mdpi.com/2076-3417/10/3/766/htm

        Uses a stack of two 1-D Convolutional layers, followed by LSTM, followed by 
        a Dense (fully connected) layer. Three preset training modes are available, 
        with further customization by editing the code. A Sequential tf.keras model 
        is implemented here.

        Note: RAM may be a limiting factor for the parameter "input_size". The wav data
          is preprocessed and stored in RAM, which improves training speed but quickly runs out
          if using a large number for "input_size".  Reduce this if you are experiencing
          RAM issues. 
        
        --training_mode=0   Speed training (default)
        --training_mode=1   Accuracy training
        --training_mode=2   Extended training (set max_epochs as desired, for example 50+)
    '''

    name = args.name
    if not os.path.exists('models/'+name):
        os.makedirs('models/'+name)
    else:
        print("A model folder with the same name already exists. Please choose a new name.")
        return

    train_mode = args.training_mode     # 0 = speed training, 
                                        # 1 = accuracy training 
                                        # 2 = extended training
    batch_size = args.batch_size 
    test_size = 0.2
    epochs = args.max_epochs
    input_size = args.input_size

    # TRAINING MODE
    if train_mode == 0:         # Speed Training
        learning_rate = 0.01 
        conv1d_strides = 12    
        conv1d_filters = 16
        hidden_units = 36
    elif train_mode == 1:       # Accuracy Training (~10x longer than Speed Training)
        learning_rate = 0.01 
        conv1d_strides = 4
        conv1d_filters = 36
        hidden_units= 64
    else:                       # Extended Training (~60x longer than Accuracy Training)
        learning_rate = 0.0005 
        conv1d_strides = 3
        conv1d_filters = 36
        hidden_units= 96


    # Load and Preprocess Data ###########################################
    in_rate, in_data = wavfile.read(args.in_file)
    out_rate, out_data = wavfile.read(args.out_file)

    X = in_data.astype(np.float32).flatten()  
    X = normalize(X).reshape(len(X),1)   
    y = out_data.astype(np.float32).flatten() 
    y = normalize(y).reshape(len(y),1)   

    y_ordered = y[input_size-1:] 

    indices = np.arange(input_size) + np.arange(len(X)-input_size+1)[:,np.newaxis] 
    X_ordered = tf.gather(X,indices) 

    shuffled_indices = np.random.permutation(len(X_ordered)) 
    X_random = tf.gather(X_ordered,shuffled_indices)
    y_random = tf.gather(y_ordered, shuffled_indices)

    # Create Sequential Model ###########################################
    clear_session()
    model = Sequential()
    model.add(Conv1D(conv1d_filters, 12,strides=conv1d_strides, activation=None, padding='same',input_shape=(input_size,1)))
    model.add(Conv1D(conv1d_filters, 12,strides=conv1d_strides, activation=None, padding='same'))
    model.add(LSTM(hidden_units))
    model.add(Dense(1, activation=None))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=error_to_signal, metrics=[error_to_signal])
    print(model.summary())

    # Train Model ###################################################
    model.fit(X_random,y_random, epochs=epochs, batch_size=batch_size, validation_split=test_size)    

    model.save('models/'+name+'/'+name+'.h5')

    # Run Prediction #################################################
    print("Running prediction..")
    y_the_rest, y_last_part = np.split(y_ordered, [int(len(y_ordered)*.8)])
    x_the_rest, x_last_part = np.split(X, [int(len(X)*.8)])

    x_the_rest, x_ordered_last_part = np.split(X_ordered, [int(len(X_ordered)*.8)])
    prediction = model.predict(x_ordered_last_part, batch_size=batch_size)

    save_wav('models/'+name+'/y_pred.wav', prediction)
    save_wav('models/'+name+'/x_test.wav', x_last_part)
    save_wav('models/'+name+'/y_test.wav', y_last_part)

    # Add additional data to the saved model (like input_size)
    filename = 'models/'+name+'/'+name+'.h5'
    f = h5py.File(filename, 'a')
    grp = f.create_group("info")
    dset = grp.create_dataset("input_size", (1,), dtype='int16')
    dset[0] = input_size
    f.close()

    # Create Analysis Plots ###########################################
    if args.create_plots == 1:
        print("Plotting results..")
        import plot

        plot.analyze_pred_vs_actual({   'output_wav':'models/'+name+'/y_test.wav',
                                            'pred_wav':'models/'+name+'/y_pred.wav', 
                                            'input_wav':'models/'+name+'/x_test.wav',
                                            'model_name':name,
                                            'show_plots':1,
                                            'path':'models/'+name
                                        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    parser.add_argument("name")
    parser.add_argument("--training_mode", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--create_plots", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=100)
    args = parser.parse_args()
    main(args)