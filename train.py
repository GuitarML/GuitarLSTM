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
          RAM issues. Also, you can use the "--split_data" option to divide the data by the
          specified amount and train the model on each set. Doing this will allow for a higher
          input_size setting (more accurate results).
        
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

    # Create Sequential Model ###########################################
    clear_session()
    model = Sequential()
    model.add(Conv1D(conv1d_filters, 12,strides=conv1d_strides, activation=None, padding='same',input_shape=(input_size,1)))
    model.add(Conv1D(conv1d_filters, 12,strides=conv1d_strides, activation=None, padding='same'))
    model.add(LSTM(hidden_units))
    model.add(Dense(1, activation=None))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=error_to_signal, metrics=[error_to_signal])
    print(model.summary())

    # Load and Preprocess Data ###########################################
    in_rate, in_data = wavfile.read(args.in_file)
    out_rate, out_data = wavfile.read(args.out_file)

    X_all = in_data.astype(np.float32).flatten()  
    X_all = normalize(X_all).reshape(len(X_all),1)   
    y_all = out_data.astype(np.float32).flatten() 
    y_all = normalize(y_all).reshape(len(y_all),1)   

    # If splitting the data for training, do this part
    if args.split_data > 1:
        num_split = len(X_all) // args.split_data
        X = X_all[0:num_split*args.split_data]
        y = y_all[0:num_split*args.split_data]
        X_data = np.split(X, args.split_data)
        y_data = np.split(y, args.split_data)

        # Perform training on each split dataset
        for i in range(len(X_data)):
            print("\nTraining on split data " + str(i+1) + "/" +str(len(X_data)))
            X_split = X_data[i]
            y_split = y_data[i]

            y_ordered = y_split[input_size-1:] 

            indices = np.arange(input_size) + np.arange(len(X_split)-input_size+1)[:,np.newaxis] 
            X_ordered = tf.gather(X_split,indices) 

            shuffled_indices = np.random.permutation(len(X_ordered)) 
            X_random = tf.gather(X_ordered,shuffled_indices)
            y_random = tf.gather(y_ordered, shuffled_indices)

            # Train Model ###################################################
            model.fit(X_random,y_random, epochs=epochs, batch_size=batch_size, validation_split=0.2)  
 

        model.save('models/'+name+'/'+name+'.h5')

    # If training on the full set of input data in one run, do this part
    else:
        y_ordered = y_all[input_size-1:] 

        indices = np.arange(input_size) + np.arange(len(X_all)-input_size+1)[:,np.newaxis] 
        X_ordered = tf.gather(X_all,indices) 

        shuffled_indices = np.random.permutation(len(X_ordered)) 
        X_random = tf.gather(X_ordered,shuffled_indices)
        y_random = tf.gather(y_ordered, shuffled_indices)

        # Train Model ###################################################
        model.fit(X_random,y_random, epochs=epochs, batch_size=batch_size, validation_split=test_size)    

        model.save('models/'+name+'/'+name+'.h5')

    # Run Prediction #################################################
    print("Running prediction..")

    # Get the last 20% of the wav data to run prediction and plot results
    y_the_rest, y_last_part = np.split(y_all, [int(len(y_all)*.8)])
    x_the_rest, x_last_part = np.split(X_all, [int(len(X_all)*.8)])
    y_test = y_last_part[input_size-1:] 
    indices = np.arange(input_size) + np.arange(len(x_last_part)-input_size+1)[:,np.newaxis] 
    X_test = tf.gather(x_last_part,indices) 

    prediction = model.predict(X_test, batch_size=batch_size)

    save_wav('models/'+name+'/y_pred.wav', prediction)
    save_wav('models/'+name+'/x_test.wav', x_last_part)
    save_wav('models/'+name+'/y_test.wav', y_test)

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
    parser.add_argument("--split_data", type=int, default=1)
    args = parser.parse_args()
    main(args)