import uproot as up
import numpy as np
import h5py
import pandas as pd
from variable_mapping import mapping

input_file = "/afs/cern.ch/work/c/cainswor/public/DL/tt_jets_NN_input.root"
Signal = up.open(input_file)['NN_signal']
Background = up.open(input_file)['NN_background']

var_list = list(mapping.keys())

df_S = Signal.pandas.df(var_list)
df_B = Background.pandas.df(var_list)

'''
append S and B
shuffule
'''


outputfile = h5py.File('path_to_output','w')
outputfile.create_dataset(X_train, data='X_train')
outputfile.close()

