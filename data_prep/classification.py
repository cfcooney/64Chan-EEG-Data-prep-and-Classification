"""
Name: Ciaran Cooney
Date: 17/08/2019
Description: Class for training CNNs using a nested cross-validation method. Train on the inner_fold to obtain
optimized hyperparameters. Train outer_fold to obtain classification performance.
"""

from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.torch_ext.util import set_random_seeds, np_to_var, var_to_np
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.functions import square, safe_log
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode.models.eegnet import EEGNetv4
from sklearn.model_selection import train_test_split
from utils import current_acc, current_loss
from torch.nn.functional import elu
from torch.nn.functional import nll_loss
from experiment import Experiment
import pandas as pd
import numpy as np
import itertools as it
import torch
from torch import optim
import logging
torch.backends.cudnn.deterministic = True

from metrics import cross_entropy

log = logging.getLogger(__name__)


class Classification():

    def __init__(self, hyp_params, parameters, data_params, subject, model_type, data_type):
        self.subject = subject
        self.model_type = model_type
        self.data_type = data_type
        self.best_loss = parameters["best_loss"]
        self.batch_size = parameters["batch_size"]
        self.monitors = parameters["monitors"]
        self.cuda = parameters["cuda"]
        self.model_constraint = parameters["model_constraint"]
        self.max_increase_epochs = parameters['max_increase_epochs']
        self.n_classes = data_params["n_classes"]
        self.n_chans = data_params["n_chans"]
        self.input_time_length = data_params["input_time_length"]
        self.hyp_params = hyp_params
        self.activation = "elu"
        self.learning_rate = 0.01
        self.epochs = 1
        self.loss = nll_loss
        for key in hyp_params:
            setattr(self, key, hyp_params[key])
        self.iterator = BalancedBatchSizeIterator(batch_size=self.batch_size)
        self.best_params = None
        self.model_number = 1
        self.y_pred = np.array([])
        self.probabilities = np.array([])

    def call_model(self):
        if self.model_type == 'shallow':
            model = ShallowFBCSPNet(in_chans=self.n_chans, n_classes=self.n_classes, input_time_length=self.input_time_length,
                                    n_filters_time=40, filter_time_length=25, n_filters_spat=40,
                                    pool_time_length=75, pool_time_stride=15, final_conv_length='auto',
                                    conv_nonlin=getattr(torch.nn.functional, self.activation), pool_mode='mean', pool_nonlin=safe_log,
                                    split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1,
                                    drop_prob=0.1).create_network()

        elif self.model_type == 'deep':
            model = Deep4Net(in_chans=self.n_chans, n_classes=self.n_classes, input_time_length=self.input_time_length,
                             final_conv_length='auto', n_filters_time=25, n_filters_spat=25, filter_time_length=10,
                             pool_time_length=3, pool_time_stride=3, n_filters_2=50, filter_length_2=10,
                             n_filters_3=100, filter_length_3=10, n_filters_4=200, filter_length_4=10,
                             first_nonlin=getattr(torch.nn.functional, self.activation), first_pool_mode='max', first_pool_nonlin=safe_log,
                             later_nonlin=self.getattr(torch.nn.functional, self.activation),
                             later_pool_mode='max', later_pool_nonlin=safe_log, drop_prob=0.1,
                             double_time_convs=False, split_first_layer=False, batch_norm=True, batch_norm_alpha=0.1,
                             stride_before_pool=False).create_network()

        elif self.model_type == 'eegnet':
            model = EEGNetv4(in_chans=self.n_chans, n_classes=self.n_classes, final_conv_length='auto',
                             input_time_length=self.input_time_length, pool_mode='mean', F1=16, D=2, F2=32,
                             kernel_length=64, third_kernel_size=(8, 4), conv_nonlin=getattr(torch.nn.functional, self.activation), drop_prob=0.1).create_network()
        return model

    def set_best_params(self):
        assert type(self.best_params) is list, "list of selected parameters required"
        for i in range(len(self.hyp_params)):
            setattr(self, list(self.hyp_params.keys())[i], self.best_params[i])

    def concat_y_pred(self, y_pred_fold):
        """
        Method for combining all outer-fold ground-truth values.
        :param y_pred_fold: array of single-fold true values.
        :return: all outer fold true values in single arrau
        """
        self.y_pred = np.concatenate((self.y_pred, np.array(y_pred_fold)))

    def concat_probabilities(self, probabilities_fold):
        """
        Method for combining all outer-fold ground-truth values.
        :param y_pred_fold: array of single-fold true values.
        :return: all outer fold true values in single arrau
        """
        self.probabilities = np.concatenate((self.probabilities, probabilities_fold))

    def train_model(self, train_set, val_set, test_set, save_model):
        """
        :param train_set: EEG data (n_trials*n_channels*n_samples)
        :param val_set: EEG data (n_trials*n_channels*n_samples)
        :param test_set: EEG data (n_trials*n_channels*n_samples) - can be None when training on inner-fold
        :param save_model: Boolean: True if trained model is to be saved
        :return: Accuracy and loss scores for the model trained with a given set of hyper-parameters
        """
        predictions = None
        model = None
        model = self.call_model()

        set_random_seeds(seed=20190629, cuda=self.cuda)

        if self.cuda:
            model.cuda()
            torch.backends.cudnn.deterministic = True

        log.info("%s model: ".format(str(model)))
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=0, eps=1e-8, amsgrad=False)
        stop_criterion = Or([MaxEpochs(self.epochs),
                             NoDecrease('valid_misclass', self.max_increase_epochs)])

        model_loss_function = None

        #####Setup to run the selected model#####
        model_test = Experiment(model, train_set, val_set, test_set=test_set, iterator=self.iterator,
                                loss_function=self.loss, optimizer=optimizer,
                                model_constraint=self.model_constraint, monitors=self.monitors,
                                stop_criterion=stop_criterion, remember_best_column='valid_misclass',
                                run_after_early_stop=True, model_loss_function=model_loss_function, cuda=self.cuda,
                                data_type=self.data_type, model_type=self.model_type, subject_id=self.subject,
                                model_number=str(self.model_number), save_model=save_model)
        model_test.run()

        model_acc = model_test.epochs_df['valid_misclass'].astype('float')
        model_loss = model_test.epochs_df['valid_loss'].astype('float')
        current_val_acc = 1 - current_acc(model_acc)
        current_val_loss = current_loss(model_loss)

        test_accuracy = None
        if test_set is not None:
            test_accuracy = round((1 - model_test.epochs_df['test_misclass'].min()) * 100, 3)
            predictions = model_test.predictions
        probabilities = model_test.probabilites

        return current_val_acc, current_val_loss, test_accuracy, model_test, predictions, probabilities

    def train_inner(self, train_set, val_set, test_set, save_model):
        val_acc, val_loss, val_cross_entropy = [], [], []
        names = list(self.hyp_params.keys())
        hyp_param_combs = it.product(*(self.hyp_params[Name] for Name in names))
        for hp_combination in hyp_param_combs:

            for i in range(len(self.hyp_params)):
                setattr(self, list(self.hyp_params.keys())[i], hp_combination[i])

            current_val_acc, current_val_loss, _, _, _, probabilities = self.train_model(train_set, val_set, test_set, save_model)
            val_acc.append(current_val_acc)
            val_loss.append(current_val_loss)
            #print(np.array(probabilities).shape)
            for dist in probabilities:
                #print(f"val_set: {val_set.y.shape} : distribution: {np.array(dist).shape}")
                val_cross_entropy.append(cross_entropy(val_set.y, dist)) #1 CE value per-HP, repeat for n_folds
        #print(val_cross_entropy)
        return val_acc, val_loss, val_cross_entropy

    def train_outer(self, trainsetlist, testsetlist, save_model):
        scores, all_preds, probabilities_list, outer_cross_entropy, fold_models = [],[],[],[],[]

        for train_set, test_set in zip(trainsetlist, testsetlist):
            trainset_X, valset_X, trainset_y, valset_y = train_test_split(train_set.X, train_set.y, test_size=0.2,
                                                                          shuffle=True, random_state=42,
                                                                          stratify=train_set.y)
            train_set = SignalAndTarget(trainset_X, trainset_y)
            val_set = SignalAndTarget(valset_X, valset_y)
            print(f"train set: {train_set.y.shape} : val_set: {val_set.y.shape} : test_set: {test_set.y.shape}")
            _, _, test_accuracy, optimised_model, predictions, probabilities = self.train_model(train_set, val_set, test_set, save_model)

            fold_models.append(optimised_model)
            probs_array = []
            for lst in probabilities:
                for trial in lst:
                    probs_array.append(trial) # all probabilities for this test-set

            print(f"/"*20)
            scores.append(test_accuracy)
            self.concat_y_pred(predictions)

            probabilities_list.append(probs_array) #outer probabilities to be used for cross-entropy
            self.model_number += 1

        for y_true, y_probs in zip(testsetlist, probabilities_list):
            outer_cross_entropy.append(cross_entropy(y_true.y, y_probs))

        return scores, fold_models, self.y_pred, probabilities_list, outer_cross_entropy





