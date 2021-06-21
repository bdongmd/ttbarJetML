import uproot as up
import numpy as np
import h5py
import pandas as pd
from variable_mapping import mapping
from keras.utils import np_utils
import plot_library

import argparse

parser = argparse.ArgumentParser(description='options for converting files')
parser.add_argument('-i', '--inputfile', type=str,
		default = '/afs/cern.ch/work/c/cainswor/public/DL/tt_jets_NN_input.root')
parser.add_argument('-o', '--outputfile', type=str)
args = parser.parse_args()

Signal = up.open(args.inputfile)['NN_signal']
Background = up.open(args.inputfile)['NN_background']

var_list = list(mapping.keys())

df_S = Signal.pandas.df(var_list)
df_B = Background.pandas.df(var_list)

df_S['label'] = pd.Series(np.ones(len(df_S), dtype=int), index=df_S.index)
df_B['label'] = pd.Series(np.zeros(len(df_B), dtype=int), index=df_B.index)
plot_library.variable_plotting(df_S, df_B)

X_train = np.concatenate((df_B, df_S))
y_train = np.concatenate((np.zeros(len(df_B)), np.zeros(len(df_S))))
Y_train = np_utils.to_categorical(y_train, 2)

rng_state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(rng_state)
np.random.shuffle(Y_train)
#assert X_train.shape[0] == len(var_list)


outputfile = h5py.File(args.outputfile,'w')
outputfile.create_dataset(X_train, data='X_train', compression='gzip')
outputfile.create_dataset(Y_train, data='Y_train', compression='gzip')
outputfile.close()

