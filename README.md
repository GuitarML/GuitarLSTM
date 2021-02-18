# GuitarLSTM

GuitarLSTM trains guitar effect/amp neural network models for processing
on wav files.  Record input/output samples from the target guitar amplifier or
pedal, then use this code to create a deep learning model of the
sound. The model can then be applied to other wav files to make it sound
like the amp or effect. This code uses Tensorflow/Keras.

The LSTM (Long short-term memory) model is effective for copying the sound of 
tube amplifiers, distortion, overdrive, and compression. It also captures the 
impluse response of the mic/cab used for recording the samples. In comparison
to the WaveNet model, this implementation is much faster and can more accurately 
copy the sound of complex guitar signals while still training on a CPU.


## Info
A variation on the LSTM model from [Real-Time Guitar Amplifier Emulation with Deep
Learning](https://www.mdpi.com/2076-3417/10/3/766/htm)


For a great explanation of how LSTMs work, check out this [blog post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).

## Data

`data/ts9_test1_in_FP32.wav` - Playing from a Fender Telecaster, bridge pickup, max tone and volume<br>
`data/ts9_test1_out_FP32.wav` - Split with JHS Buffer Splitter to Ibanez TS9 Tube Screamer
(max drive, mid tone and volume).<br>
`models/ts9_model.h5` - Pretrained model weights


## Usage

**Train model and run effect on .wav file**:
Must be single channel, 44.1 kHz, FP32 wav data (not int16)
```bash
# Preprocess the input data, perform training, and generate test wavs and analysis plots. 
# Specify input wav file, output wav file, and desired model name.
# Output will be saved to "models/out_model_name/" folder.

python train.py data/ts9_test1_in_FP32.wav data/ts9_test1_out_FP32.wav out_model_name


# Run prediction on target wav file
# Specify input file, desired output file, and model path
python predict.py data/ts9_test1_in_FP32.wav output models/ts9_model.h5
```

**Training parameters**:

```bash
# Use these arguments with train.py to further customize the model:

--training_mode=0  # enter 0, 1, or 2 for speed tranining, accuracy training, or extended training, respectively
--input_size=150   # sets the number of previous samples to consider for each output sample of audio
--split_data=3     # splits the input data by X amount to reduce RAM usage; trains the model on each split separately
--max_epochs=1     # sets the number of epochs to train for; intended to be increased dramatically for extended training
--batch_size=4096  # sets the batch size of data for training

# Edit the "TRAINING MODE" or "Create Sequential Model" section of train.py to further customize each layer of the neural network.
```

**Colab Notebook**:
Use Google Colab notebook (guitar_lstm_colab.ipynb) for training 
GuitarLSTM models in the cloud. See notebook comments for instructions.

## Training Info

Helpful tips on training models:
1. Wav files should be 3 - 4 minutes long, and contain a variety of
   chords, individual notes, and playing techniques to get a full spectrum
   of data for the model to "learn" from.
2. A buffer splitter was used with pedals to obtain a pure guitar signal
   and post amp/effect signal. You can also use a feedback loop from your
   audio interface to record input/output simultaneously.
3. Obtaining sample data from an amp can be done by splitting off the original
   signal, with the post amp signal coming from a microphone (I used a SM57).
   Keep in mind that this captures the dynamic response of the mic and cabinet.
   In the original research the sound was captured directly from within the amp
   circuit to have a "pure" amp signal.
4. Generally speaking, the more distorted the effect/amp, the more difficult it
   is to train. 
5. Requires float32 .wav files for training (as opposed to int16).
   
   
## Limitations and future work

This implementation of the LSTM model uses a high amount of
RAM to preprocess wav data. If you experience crashes due to 
limited memory, reduce the "input_size" parameter by using 
the "--input_size=" flag with train.py. The default setting is 100,
which requires about 8GB of RAM. Increasing this setting will improve 
training accuracy, but the size of the preprocessed wav data in 
RAM will increase as well.

You can also use the "--split_data" parameter with train.py to
train the same model on separate sections of the data. This
will reduce RAM usage while still allowing a high input_size
setting. For example, "--split_data=5" would split the data 
into 5 sections, and train each section separately. The default
is 1, or no splitting.

A custom dataloader has been added to the Colab notebook using MSE
for the loss calculation. This reduces RAM usage and eliminates the 
need for the --split_data parameter.
   
A real-time implementation for use in a guitar plugin: [SmartAmpPro](https://github.com/GuitarML/SmartAmpPro)

Note: The model training has been integrated into the SmartAmpPro plugin, the 
models trained with GuitarLSTM are not currently compatible with the plugin.
