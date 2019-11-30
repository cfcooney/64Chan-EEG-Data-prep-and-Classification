# 64Chan-EEG-Data-prep-and-Classification
Jupyter Notebooks and Python scripts for processing, analysing and classifying EEG data recorded from a 64-channel data acquisition system while subjects performed overt and imagined speech tasks.

- preprocessing/preprocessing_main.ipynb: *Read in EEG data and set topographical montage 
                                          *Create MNE raw array, plot and view raw EEG
                                          *Validate channels, remove flat or noisy channels
                                          *Bandpass filter, rereference and apply baseline correction
                                          *Epoch the data, view trials and remove bad trials
                                          *Perform Independent Component Analysis for artefact removal
                                          
- classification.py: *Class for training CNNs using a nested cross-validation method. Train on the inner_fold to obtain
optimized hyperparameters. Train outer_fold to obtain classification performance.

- results.py: *Class for computing and stroing results for individual subjects. Includes accuracy, f1-score, etc. *inner- and outer-fold scores are saved and predicted labels stored for future use.
