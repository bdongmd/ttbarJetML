import h5py
import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(
    description = 'Options for making training and testing loss plots'
)

parser.add_argument('-i', '--input_file', type=str,
    default = 'results/r12573_noDropout_b3000_e150/model_epochs150_r12573_loss.h5',
    help = 'Set name of input file'
)

parser.add_argument('-o', '--output_directory', type=str,
    default = 'plots/',
    help = 'Set name of input file'
)

args = parser.parse_args()

fin = h5py.File(args.input_file, 'r')

trainLoss = fin['train_loss'][:]
testLoss = fin['val_loss'][:]
trainAcc = fin['train_acc'][:]
testAcc = fin['val_acc'][:]

epochs = np.arange(1, len(trainLoss) + 1)

fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
plt.plot(epochs, trainLoss, 'o')
plt.plot(epochs, testLoss, 'o')
plt.legend(['training loss', 'testing loss'], loc='upper right')
plt.ylabel('loss')
plt.yscale('log')
plt.xlabel('epoch')
#plt.ylim(0.5, 0.9)
plt.savefig('{}/loss_compare.pdf'.format(args.output_directory))

fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
plt.plot(epochs, trainAcc, 'o')
plt.plot(epochs, testAcc, 'o')
plt.legend(['training accuracy', 'testing accuracy'], loc='upper right')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.ylim(0.6, 0.9)
plt.savefig('{}/acc_compare.pdf'.format(args.output_directory))
