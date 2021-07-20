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
h5f_train = h5py.File(args.input_file, 'r')

totalEvents = len(h5f_train['X_train'])
trainEvents = int(0.8*totalEvents)

X_train = h5f_train['X_train'][:trainEvents]
Y_train = h5f_train['Y_train'][:trainEvents]
X_test = h5f_train['X_train'][trainEvents:]
Y_test = h5f_train['Y_train'][trainEvents:]


InputShape=31
h_layers=[60, 30, 15, 8]
lr = 0.005
drops=[0.1, 0.2, 0.2, 0.2]
dropout=True
batch_size = args.batch_size

Model = model.private_DL1Model(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops, dropout=dropout)
Model.summary()

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
                    callbacks=callbacks
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
