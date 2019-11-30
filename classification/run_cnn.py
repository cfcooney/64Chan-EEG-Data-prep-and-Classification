from subjects import Subject
from results import Results, CombinedResults 
from utils import windows
from utils import load_subject
from braindecode.datautil.signal_target import SignalAndTarget
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import pandas as pd
import logging
import time
import sys


hyp_params = dict(activation = ['elu', 'relu6', 'leaky_relu', 'relu'],
                  lr=[0.001,0.01,0.1,1],
                  epochs=[20,40,60,80])

direct  = 'C:/Users/cfcoo/OneDrive - Ulster University/Study_3/Subject_Data'
subject = '01'
subj = load_subject(direct, subject, 1, "EEG_semantics_text")["subject"]


data   = subj.data3D.astype(np.float32)
labels = subj.labels.astype(np.int64)

unique = np.unique(labels, return_counts=False)
n_classes = len(unique)
n_chans   = subj.n_chans
input_time_length = subj.epoch

w = windows(data, subj, 500, 250, 500) # fs = subj.sfreq

num_folds = 4
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=10)
out_fold_num = 0 # outer-fold number
trainsetlist, testsetlist = [],[]

subj_results = Results(subject, num_folds, class_names=["apple","orange","car","bus"])
subj_results.get_acc_loss_df(hyp_params, 'Fold')


for inner_ind, outer_index in skf.split(data, labels):
    inner_fold, outer_fold     = data[inner_ind], data[outer_index]
    inner_labels, outer_labels = labels[inner_ind], labels[outer_index]
    subj_results.concat_y_true(outer_labels)

    out_fold_num += 1
    in_fold_num = 0

    trainsetlist.append(SignalAndTarget(inner_fold, inner_labels)) #used for outer-fold train/test
    testsetlist.append(SignalAndTarget(outer_fold, outer_labels))

    for train_idx, valid_idx in skf.split(inner_fold, inner_labels):
        X_Train, X_val = inner_fold[train_idx], inner_fold[valid_idx]
        y_train, y_val = inner_labels[train_idx], inner_labels[valid_idx]
        train_set = SignalAndTarget(X_Train, y_train)
        val_set = SignalAndTarget(X_val, y_val)
        in_fold_num += 1
        hyp_param_acc, hyp_param_loss = [], []
        
        #hyp_param_loss, hyp_param_acc = train_inner(train_set, val_set,hyp_params,parameters)


