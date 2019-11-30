"""
Name: Ciaran Cooney
Date: 28/08/2019
Description: Main file for running nested cross-validated training and testing of CNNs on
imagined speech EEG data. Inner-fold used to select hyper-parameters. Outer-fold for final 
model and testing. A Results object is created for each subject and scores related to each
saved as Excel files to a local drive.
"""

from subjects import Subject
from results import Results
from classification import Classification
from utils import load_subject, format_data, timer, windows, get_model_loss_and_acc
from braindecode.datautil.signal_target import SignalAndTarget
from sklearn.model_selection import StratifiedKFold
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, RuntimeMonitor
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


hyp_params = dict(activation=["elu","relu"],
                  lr=[0.001],
                  epochs=[1,2])
parameters = dict(best_loss = 100.0,
                  batch_size = 64,
                  monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()],
                  model_constraint = MaxNormDefaultConstraint(),
                  max_increase_epochs = 1,
                  cuda = False)

@timer
def trainNestedCV(direct, subject, session, filename, hyp_params, parameters):


  subj = load_subject(direct, subject, 1, filename)["subject"]
  #
  # data = subj.data3D.astype(np.float32) # convert data to 3d for deep learning
  # labels = subj.labels.astype(np.int64)
  # labels[:] = [x - 1 for x in labels]
  data, labels = format_data('words', subject, 4096)

  import random #just for testing
  labels = [] #just for testing
  for i in range(200): #just for testing
    labels.append(random.randint(0,3)) #just for testing

  labels = np.array(labels).astype(np.int64)
  data = data[:200,:,0:750]

  unique = np.unique(labels, return_counts=False)
  data_params = dict(n_classes = len(unique),
                     n_chans=6,
                     input_time_length = subj.epoch) #n_chans = subj.n_chans

  #w = windows(data, subj, 500, 250, 500)  # fs = subj.sfreq # list of windows
  
  num_folds = 2
  skf = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=10) # don't randomize trials to preserce structure
 
  trainsetlist, testsetlist = [], []
  inner_fold_acc,inner_fold_loss, inner_fold_CE = [],[],[]
  
  subj_results = Results(subject, filename, num_folds) #, class_names=["apple", "orange", "car", "bus"]
  subj_results.change_directory(direct)

  subj_results.get_acc_loss_df(hyp_params, 'Fold') # empty dataframe headed with each HP set

  clf = Classification(hyp_params, parameters, data_params, "01", "shallow", "words") # classifier object

  print(f"Inner-fold training for Subject {subject} in progress...")
  for inner_ind, outer_index in skf.split(data, labels):
      inner_fold, outer_fold = data[inner_ind], data[outer_index]
      inner_labels, outer_labels = labels[inner_ind], labels[outer_index]
      subj_results.concat_y_true(outer_labels)


      trainsetlist.append(SignalAndTarget(inner_fold, inner_labels))  # used for outer-fold train/test
      testsetlist.append(SignalAndTarget(outer_fold, outer_labels))

      for train_idx, valid_idx in skf.split(inner_fold, inner_labels):
          X_Train, X_val = inner_fold[train_idx], inner_fold[valid_idx]
          y_train, y_val = inner_labels[train_idx], inner_labels[valid_idx]
          train_set = SignalAndTarget(X_Train, y_train)
          val_set = SignalAndTarget(X_val, y_val)
          
          hyp_param_acc, hyp_param_loss = [], []
          hyp_param_acc, hyp_param_loss, hyp_param_CE = clf.train_inner(train_set, val_set, None, False)

          inner_fold_loss.append(hyp_param_loss)
          inner_fold_acc.append(hyp_param_acc)
          inner_fold_CE.append(hyp_param_CE)

  subj_results.fill_acc_loss_df(inner_fold_acc, inner_fold_loss, inner_fold_CE)

  subj_results.get_hp_means(hyp_params, "accuracy") #needed to select inter-subject parameters

  subj_results.get_best_params("accuracy")
  clf.best_params = subj_results.best_params
  clf.set_best_params()
  print(f"Best parameters selected: {clf.best_params}")
  print("///////-------------------------------------------------------///////")
  print(f"Outer-fold training and testing for Subject {subject} in progress...")
  scores, fold_models, predictions, probabilities, outer_cross_entropy = clf.train_outer(trainsetlist, testsetlist, False) #accuracy score for each fold, combined predictions for each fold

  subj_results.outer_fold_accuracies = scores
  subj_results.y_pred= np.array(predictions)
  subj_results.y_probs = np.array(probabilities)
  subj_results.outer_fold_cross_entropies = outer_cross_entropy

  subj_results.train_loss, subj_results.valid_loss, subj_results.test_loss, subj_results.train_acc, subj_results.valid_acc, subj_results.test_acc  = get_model_loss_and_acc(fold_models)

  subj_results.save_result()

  subj_results.subject_stats()
  print("")
  print(subj_results.subject_stats_df.head())



if __name__ == '__main__':

  direct = 'C:/Users/cfcoo/OneDrive - Ulster University/Study_3/Subject_Data'
  subjects = ['02']
  session = 2
  filename = "EEG_semantics_text"
  hyp_param_means_list = []

  for subject in subjects:
    trainNestedCV(direct, subject, session, filename, hyp_params, parameters)




  

