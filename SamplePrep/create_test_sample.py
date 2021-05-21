import numpy as np
import h5py
import matplotlib.pyplot as plt
import json
import pandas as pd
import process_tool 
import plot_library
from keras.utils import np_utils

parser = argparse.ArgumentParser(description='options for making plots')
parser.add_argument('-i', '--inputfile', type=str,
		default="/eos/user/b/bdong/FTAGUpgrade/hdf5/btag_itk_r12573/MC15-600012-ttbar-r12573-alljets-even.h5",
		help='Set name of input file')
parser.add_argument('-o', '--outputfile', type=str,
		default="/eos/user/b/bdong/FTAGUpgrade/hdf5/btag_itk_r12573/MC15-600012-ttbar-r12573-test_sample_NN.h5",
		help='Set name of output file')
args = parser.parse_args()


with open("DL1_Variables.json") as vardict:
	var_names = json.load(vardict)[:]
if 'HadronConeExclTruthLabelID' in var_names:
	var_names.remove('HadronConeExclTruthLabelID')

def GetTestSample(jets):
    with open("config/params-MC15-ttbar-r12573-selection.json",'r') as infile:
        scale_dict = json.load(infile)

    jets = pd.DataFrame(jets)
    jets.query('HadronConeExclTruthLabelID<=5', inplace=True)
    jets_pt_eta = jets[['pt_btagJes', 'absEta_btagJes']]
    labels = jets['HadronConeExclTruthLabelID'].values
    jets = jets[var_names]    
    jets.replace([np.inf, -np.inf], np.nan, inplace=True)
    default_dict = process_tool.Gen_default_dict(scale_dict)
    jets.fillna(default_dict, inplace=True)

    for elem in scale_dict:
        if 'isDefaults' in elem['name']:
            continue
        if elem['name'] not in var_names:
            continue
        else:
            jets[elem['name']] = ((jets[elem['name']] - elem['shift']) / elem['scale'])

    labels_cat = np.copy(labels)
    labels_cat[labels_cat==5] = 2
    labels_cat[labels_cat==4] = 1
    labels_cat = np_utils.to_categorical(labels_cat, 3)

    return jets.values, jets_pt_eta.to_records(index=False), labels, labels_cat

df_tt_test = h5py.File(args.inputfile, "r")['jets'][:]
X_test, jpt, labels, Y_test = GetTestSample(df_tt_test)
rng_state = np.random.get_state()
np.random.shuffle(X_test)
np.random.set_state(rng_state)
np.random.shuffle(Y_test)
np.random.set_state(rng_state)
np.random.shuffle(labels)

h5f = h5py.File(args.outputfile, 'w')
h5f.create_dataset('X_test', data=X_test, compression='gzip')
h5f.create_dataset('Y_test', data=Y_test, compression='gzip')
#h5f.create_dataset('pt_eta', data=jpt, compression='gzip')
h5f.create_dataset('labels', data=labels, compression='gzip')
h5f.close()
