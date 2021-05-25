import uproot as up
import numpy as np
import h5py
import pandas as pd
from variable_mapping import mapping

input_file = "path_to_file"
Signal = up.open(input_file)['Signal']
Background = up.open(input_file)['Background']

df_S = Signal.pandas.df(var_list)
df_B = Background.pandas.df(var_list)

'''
append S and B
shuffule
'''


outputfile = h5py.File('path_to_output','w')
outputfile.create_dataset(X_train, data='X_train')
outputfile.close()

