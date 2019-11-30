from results import CombinedResults
from viz_utils import print_confusion_matrix, get_roc, combine_subjects, plot_acc_or_loss, get_mean, scattered_boxplot
from utils import get_class_labels, load_pickle
import numpy as np
import pandas as pd

direct = 'C:/Users/sb00745777/OneDrive - Ulster University/Study_3/Subject_Data'
subjects = ["01", "02"]
paradigm = "EEG_semantics_text"
filename = "combineResults"
identifier="all_subjects"
n_folds = 2

# Load pre-compiled Results object containing all combined-subject results
total_results, _ = load_pickle(f"{direct}/results/",f"{paradigm.replace('EEG_','')}",f"{filename}.pickle")
total_results = total_results['subject']

print(f"               Combined results for paradigm: {total_results.paradigm.replace('EEG_','')}")
print(total_results.subject_stats_df.head())


print_confusion_matrix(total_results.cm, labels, f"C:/Users/sb00745777/OneDrive - Ulster University/Study_3/Subject_Data/results/{paradigm.replace('EEG_','')}/confusion_matrix")

get_roc(total_results.y_pred, total_results.y_true, labels,
         filename=f"{direct}/results/{paradigm.replace('EEG_','')}/roc_curve", show_plot=True)

# Plot and save losses
mean_loss_train, mean_loss_valid, mean_loss_test = get_mean(total_results, metric="loss")
mean_acc_train, mean_acc_valid, mean_acc_test = get_mean(total_results, metric="acc")
x = mean_loss_train.index.tolist()
x = [n+1 for n in x]
plot_acc_or_loss(x, mean_loss_train, mean_loss_valid, mean_loss_test, ylabel="Loss", filename=f"{direct}/results/{paradigm.replace('EEG_','')}/loss_plot", show_plot=True)
plot_acc_or_loss(x, mean_acc_train, mean_acc_valid, mean_acc_test, ylabel="Accuracy", filename=f"{direct}/results/{paradigm.replace('EEG_','')}/acc_plot", show_plot=True)


##### Draw box plots with data scattered across
inner_box_acc_data = total_results.HP_acc['Mean'].values
inner_box_loss_data = total_results.HP_loss['Mean'].values
inner_box_ce_data = total_results.HP_ce['Mean'].values

scattered_boxplot(inner_box_acc_data, xlabel=paradigm.replace('EEG_',''), ylabel='Classification Accuracy (%)',
                  filename=f"{direct}/results/{paradigm.replace('EEG_','')}/acc_scatterbox", show_plot=True)
scattered_boxplot(inner_box_loss_data, xlabel=paradigm.replace('EEG_',''), ylabel='Loss',
                  filename=f"{direct}/results/{paradigm.replace('EEG_','')}/loss_scatterbox", show_plot=True)
scattered_boxplot(inner_box_ce_data, xlabel=paradigm.replace('EEG_',''), ylabel='Cross Entropy',
                  filename=f"{direct}/results/{paradigm.replace('EEG_','')}/crossentropy_scatterbox", show_plot=True)
