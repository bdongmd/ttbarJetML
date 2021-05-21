import numpy as np
import h5py
import matplotlib.pyplot as plt
import json
import pandas as pd
import process_tool
import plot_library
from keras.utils import np_utils
from numpy.lib.recfunctions import repack_fields
from datetime import datetime

parser = argparse.ArgumentParser(description='options for making plots')
parser.add_argument('-i', '--inputfile', type=str,
		default="/eos/user/b/bdong/FTAGUpgrade/hdf5/btag_itk_r12573/MC15-600012-ttbar-r12573-alljets-odd.h5",
		help='Set name of input file')
parser.add_argument('-o', '--outputfile', type=str,
		default="/eos/user/b/bdong/FTAGUpgrade/hdf5/btag_itk_r12573/MC15-600012-ttbar-r12573-train_sample_NN.h5",
		help='Set name of output file')
args = parser.parse_args()

ttbar_file = h5py.File(args.inputfile, "r") 
jets = ttbar_file['jets']
jets = jets[:6746319]
print("{} Progress -- jets loaded.".format(datetime.now().strftime("%H:%M:%S")))

Xjets = pd.DataFrame(jets)
Xjets = Xjets.rename(index=str, columns={'HadronConeExclTruthLabelID': 'label'})
ujets = Xjets.query('label==0')
n_ujets = int(len(ujets) *1.5)
bjets = Xjets.query('label==5')[:n_ujets]
cjets = Xjets.query('label==4')[:n_ujets]
X = pd.concat([ujets, bjets, cjets])
del ujets, cjets, bjets, Xjets
print("{} Progress -- transfered input to pd.DataFrame.".format(datetime.now().strftime("%H:%M:%S")))
print("{} Progress -- data size: {}.".format(datetime.now().strftime("%H:%M:%S"), X.shape)) 

#### plot input variables
print("{} Progress -- Plotting variables before re-weight.".format(datetime.now().strftime("%H:%M:%S")))
plot_library.variable_plotting(X, outputFile = "output/DL1_variables_before-weighting.pdf")

print("{} Progress -- Reweighting.".format(datetime.now().strftime("%H:%M:%S")))
X['weight'] = process_tool.GetWeights(X)

scale_dict = []

with open("DL1_Variables.json") as vardict:
	var_names = json.load(vardict)[:]
if 'HadronConeExclTruthLabelID' in var_names:
	var_names.remove('HadronConeExclTruthLabelID')

for var in var_names:
	if var in ['label', 'weight']:
		continue
	elif 'isDefaults' in var:
		scale_dict.append(process_tool.dict_in(var, 0., 1., None))
	else:
		dict_entry = process_tool.Get_Shift_Scale(vec=X[var].values, w=X['weight'].values, varname=var)
		scale_dict.append(process_tool.dict_in(*dict_entry))

with open('config/params-MC15-ttbar-r12573-selection.json', 'w') as outfile:
	json.dump(scale_dict, outfile, indent=4)

default_dict = process_tool.Gen_default_dict(scale_dict)
X.fillna(default_dict, inplace=True)

for elem in scale_dict:
	if 'isDefaults' in elem['name']:
		continue
	else:
		X[elem['name']] = (X[elem['name']] - elem['shift']) / elem['scale']

bjets = X.query('label==5')[var_names]
cjets = X.query('label==4')[var_names]
ujets = X.query('label==0')[var_names]
bweight =  X.query('label==5')['weight']
cweight =  X.query('label==4')['weight']
uweight =  X.query('label==0')['weight']

print("{} Progress -- Plotting variables after re-weight.".format(datetime.now().strftime("%H:%M:%S")))
plot_library.variable_plotting(X, outputFile = "output/DL1_variables_after-Scaling.pdf")

print("{} Progress -- Preparing output variables.".format(datetime.now().strftime("%H:%M:%S")))
X_train = np.concatenate((ujets, cjets, bjets))
X_weight = np.concatenate((uweight, cweight, bweight))
y_train = np.concatenate((np.zeros(len(ujets)), np.ones(len(cjets)), 2*np.ones(len(bjets))))

Y_train = np_utils.to_categorical(y_train, 3)
#X_train = repack_fields(X_train[var_names])
X_train = X_train.view(np.float64).reshape(X_train.shape + (-1,))
print(X_train.shape)
print(X_weight.shape)
print(Y_train.shape)
rng_state = np.random.get_state()
np.random.shuffle(X_train)
np.random.set_state(rng_state)
np.random.shuffle(Y_train)
np.random.set_state(rng_state)
np.random.shuffle(X_weight)
assert X_train.shape[1] == len(var_names)

print("{} Progress -- Writing output file.".format(datetime.now().strftime("%H:%M:%S")))
h5f = h5py.File(args.outputfile, 'w')
h5f.create_dataset('X_train', data=X_train, compression='gzip')
h5f.create_dataset('X_weight', data=X_weight, compression='gzip')
h5f.create_dataset('Y_train', data=Y_train, compression='gzip')
h5f.close()
