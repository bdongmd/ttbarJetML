# ttbarJetML

## Dependency
- uproot=3.4.3
- pandas
- h5py
- numpy

## Instructions
- Convert root files to training file in hdf5 format:
```
cd SamplePrep
python3 convert_train.py -o training_file.h5
```
- Launch training:
```
python3 train.py -i your_input_file.h5 -o your_output_file.h5 -b batch_size -e number_of_epochs
```
- plotting
```
cd plotting
python3 plot_loss.py -i input_file.h5 -o output_directory
```
