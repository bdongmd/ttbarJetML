import imp
import h5py
import model

from keras.callbacks import EarlyStopping, ModelCheckpoint 

import plot_lib

import argparse

parser = argparse.ArgumentParser(
    description='Options for making the training files'
)
parser.add_argument('-i', '--input_file', type=str,
		default="input/MC16d_hybrid-training_sample-NN.h5",
		help='Set name of preprocessed input training file')
parser.add_argument('-o', '--output', type=str,
		default='output',
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

## selections
doPlotting = True
doSaveLoss = True

## setup model parameters
InputShape=31 ## number of input variables
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
X_test = h5f_train['X_train'][trainEvents:]
Y_train = h5f_train['labels'][:trainEvents]
Y_test = h5f_train['labels'][trainEvents:]
print("======= Progress: sample loaded")

## load model
Model = model.DNNModel(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops, dropout=dropout)
Model.summary()

## callbacks
## EarlyStopping: stop training if the validation accuracy doesn't incease for 20 epochs
## ModelCheckPoint: save th model weights with the best accuracy
callbacks = [
		EarlyStopping(verbose=True, patience=20, monitor='val_accuracy'),
		ModelCheckpoint('{}/training_b{}_e{}_bestacc.h5'.format(args.output, args.batch_size, args.epoch), monitor='val_accuracy', verbose=True, save_best_only=True, mode='max')
]

print("======= Progress: training starts")
history = Model.fit(X_train, Y_train,
                    batch_size = args.batch_size,
                    epochs = args.epoch,
                    validation_data=(X_test, Y_test),
                    callbacks=callbacks,
		    verbose=2
                    )

Model.save("{}/training_b{}_e{}.h5".format(args.output, args.batch_size, args.epoch))

train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

if doPlotting:
	plot_lib.plotAccLoss(train_loss, val_loss, putVar='Loss', output_dir=args.output)
	plot_lib.plotAccLoss(train_acc, val_acc, putVar='Acc', output_dir=args.output)

if doSaveLoss:
	hf = h5py.File('{}/train_loss.h5'.format(args.output), 'w')
	hf.create_dataset('train_loss', data=train_loss)
	hf.create_dataset('train_acc', data=train_acc)
	hf.create_dataset('val_loss', data=val_loss)
	hf.create_dataset('val_acc', data=val_acc)
	hf.close()
