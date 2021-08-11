import sys
import numpy as np
import os
import h5py
import model
import tensorflow as tf

from keras.models import load_model
from keras.utils import np_utils
import keras
import argparse

parser = argparse.ArgumentParser(
    description='Options for making the training files'
)
parser.add_argument('-i', '--input_file', type=str,
		default="input/MC16d_hybrid-training_sample-NN.h5",
		help='Set name of preprocessed input training file')

parser.add_argument('-o', '--output', type=str,
		default='output/model_predicted.h5',
		help='set the output file')
parser.add_argument('-b', '--batch_size', type=int,
		default=3000,
		help='Set batch size')
parser.add_argument('-e', '--epoch', type=int,
		default=300,
		help='Set epoch')

args = parser.parse_args()

## load input files
h5f_train = h5py.File(args.input_file, 'r')

## setup model parameters
InputShape=31
outputShape=1 ## nodes in output layer
h_layers=[60, 30, 15, 8] ## nodes in hiden layeres
lr = 0.005 ## learning rate
dropout=True ## True: use dropout in the mode. else not
drops=[0.1, 0.2, 0.2, 0.2] ## dropout probabilities in each hidden layer 
batch_size = args.batch_size

## get total number of events in the input sample
totalEvents = len(h5f_train['X_train'])
## use 80 percent for training and 20 percent for testing. The propotion can change
trainEvents = int(0.8*totalEvents)
X_train = h5f_train['X_train'][:trainEvents]

## use Y_train for 2 output, and labels for 1 output
if outputShape == 1:
	Y_train = h5f_train['labels'][:trainEvents]
	Y_test = h5f_train['labels'][trainEvents:]
elif outputShape == 2:
	Y_train = h5f_train['Y_train'][:trainEvents]
	Y_test = h5f_train['Y_train'][trainEvents:]
else:
	print("ERROR: wrong output numbers. The number of output categories can only be 1 or 2.")
	sys.exit()

## load model
Model = model.private_DL1Model(InputShape=InputShape, outputShape=outputShape, h_layers=h_layers, lr=lr, drops=drops, dropout=dropout)
Model.summary()

## callbacks, to save model for each epoch
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='models/model_{epoch}',
        save_freq='epoch'
    )
]

history = Model.fit(X_train, Y_train,
                    batch_size = args.batch_size,
                    epochs = args.epoch,
                    validation_data=(X_test, Y_test),
                    #callbacks=callbacks
                    )

Model.save("models/training_b{}_e{}.h5".format(args.batch_size, args.epoch))

train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

hf = h5py.File(args.output, 'w')
hf.create_dataset('train_loss', data=train_loss)
hf.create_dataset('train_acc', data=train_acc)
hf.create_dataset('val_loss', data=val_loss)
hf.create_dataset('val_acc', data=val_acc)
hf.close()
