# GuitarLSTM

GuitarLSTM trains guitar effect/amp neural network models for processing
on wav data.  In comparison to the WaveNet model from PedalNetRT, this
implementation is much faster and better suited for copying the sound
of guitar amps and pedals. 


## Info
Re-creation of LSTM model from [Real-Time Guitar Amplifier Emulation with Deep
Learning](https://www.mdpi.com/2076-3417/10/3/766/htm)


For a great explanation of how LSTMs works, check out this blog post:<br>
https://colah.github.io/posts/2015-08-Understanding-LSTMs/


## Data

`data/ts9_test1_in_FP32.wav` - Playing from a Fender Telecaster, bridge pickup, max tone and volume<br>
`data/ts9_test1_out_FP32.wav` - Split with JHS Buffer Splitter to Ibanez TS9 Tube Screamer
(max drive, mid tone and volume).<br>
`models/ts9_model.h5` - Pretrained model weights


## Usage

**Run effect on .wav file**:
Must be single channel, 44.1 kHz, FP32 wav data (not int16)
```bash
# This will preprocess the input data, perform training, and generate test wavs and analysis plots. 
#    Output will be saved to "models/out_model_name" folder.
python train.py data\ts9_test1_in_FP32.wav data\ts9_test1_out_FP32.wav out_model_name

# Run prediction on target wav file
# Specify input file, desired output file, and model path
predict.py data\ts9_test1_in_FP32.wav output models\ts9_model.h5
```


## Training Info

Helpful tips on training models:
1. Wav files should be 3 - 4 minutes long, and contain a variety of
   chords, individual notes, and playing techniques to get a full spectrum
   of data for the model to "learn" from.
2. A buffer splitter was used with pedals to obtain a pure guitar signal
   and post effect signal.
3. Obtaining sample data from an amp can be done by splitting off the original
   signal, with the post amp signal coming from a microphone (I used a SM57).
   Keep in mind that this captures the dynamic response of the mic and cabinet.
   In the original research the sound was captured directly from within the amp
   circuit to have a "pure" amp signal.
4. Generally speaking, the more distorted the effect/amp, the more difficult it
   is to train. 
5. Requires float32 .wav files for training (as opposed to int16).
   
   
## Future Work 
   
A real time implementation for use in a guitar plugin is
currenty in work. This would theoretically perform much faster
(less cpu usage) than the previous WaveNet model. If you want to
use deep learning models through a real time guitar plugin, 
reference the following repositories:

PedalNetRT<br>
https://github.com/GuitarML/PedalNetRT<br>

SmartGuitarAmp<br>
https://github.com/GuitarML/SmartGuitarAmp<br>


Note: See Google Colab notebook (guitar_lstm_colab.ipynb) for training 
      GuitarLSTM models in the cloud. See notebook comments for instructions.