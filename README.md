This package is used for ttbar plus c analysis to distinush signal and background using Deep neural network.

## Dependency
- uproot=3.4.3
- pandas
- h5py
- numpy
- tensorflow/keras
- matplotlib

## Instructions
- Convert root files to training file in hdf5 format:
```
cd SamplePrep
python3 convert_train.py -o training_file.h5
```
- Launch training:\
model is defined in [model.py](https://github.com/bdongmd/ttbarJetML/blob/main/model.py)

```
python3 train.py -i your_input_file.h5 -o your_output_file.h5 -b batch_size -e number_of_epochs
```

- Do evaluation (includes score plotting)
```
python3 evaluaiton.py -i testing_input_file.h5 -m trained_model.h5 -o output_directory 
```
