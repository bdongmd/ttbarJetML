import numpy as np
import h5py
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(
    description = 'options for running evaluation'
)
parser.add_argument('-i', '--input_file',type=str,
    help='file used to do evaluation'
)
parser.add_argument('-m', '--model_name',type=str,
    help='trained model'
)
parser.add_argument('-o', '--output_file',type=str,
    help='output file'
)

args = parser.parse_args()

h5f_test = h5py.File(args.input_file, 'r')

X_test = h5f_test['X_train'][:]
Y_test = h5f_test['Y_train'][:]
labels = h5f_test['labels'][:]

h5f_test.close()

test_model = tf.keras.models.load_model(args.model_name)
output = test_model.predict(X_test, verbose=2)

h5f = h5py.File(args.output_file, 'w')
h5f.create_dataset('outputScore', data=output[:,0], compression='gzip')
h5f.create_dataset('labels', data=labels, compression='gzip')
h5f.close()
