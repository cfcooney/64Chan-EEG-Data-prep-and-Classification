"""
Name: Ciaran Cooney
Date: 02/09/2019
Description: Suite of functions for plotting figures.
Includes confusion matrices and ROC curves
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp
from sklearn.utils import check_array
import pylab as P

def print_confusion_matrix(confusion_matrix, class_names, filename, normalize=True, figsize=(5, 5), fontsize=16):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    if normalize:
        confusion_matrix = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]) * 100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    fmt = '.2f' if normalize else 'd'
    #####set heatmap customization#####
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, cmap='GnBu', linewidths=.5, cbar=False,
                              annot_kws={"size": 16})
        for t in heatmap.texts:
            t.set_text(t.get_text() + "%")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Ground Truth', fontsize=16, fontname='sans-serif')
    plt.xlabel('Prediction', fontsize=16, fontname='sans-serif')

    if filename != None:
        fig.savefig(filename + '.png', bbox_inches='tight')  # store image as .png

    return fig

def add_percentages(cm):
    """
    Adds percentage signs to confusion matix values by converting to strings
    :param cm: ndarray of ints
    :return: ndarray of strings
    """
    row_data = []
    for row in cm:
        col_data = []
        for element in row:
            col_data.append(f"{element}%")
        row_data.append(col_data)

def check_int(arr):
    arr = check_array(arr, ensure_2d=False)
    if arr.dtype == np.int32:
        return arr
    else:
        return arr.astype(np.int32)

def convert_to_binary(data, classes):
    """
    Converts nurmeric results into one-hot
    encoded list for use with roc_curve.
    """

    binary_data = np.zeros((len(data), len(classes))).astype(np.int64)
    for i in classes:
        for j in range(len(data)):
            if data[j] == i:
                binary_data[j, i] = 1

    return binary_data


def roc_plot(fpr, tpr, roc_auc, unique, class_names, average="weighted", figsize=(6, 4), style='seaborn-whitegrid', fontsize='14',
             fontname='Times New Roman', filename=None, lw=1, show_plot=False):
    """
    plots ROC curve for each of the multiple classes and an average
    ROC curve. The plot is saved to specified folder.
    """

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(unique))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(unique)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(unique)

    fpr[average] = all_fpr
    tpr[average] = mean_tpr
    roc_auc[average] = auc(fpr[average], tpr[average])

    # Plot all ROC curves
    fig = plt.figure(figsize=figsize)
    plt.style.use(style)

    plt.plot(fpr["weighted"], tpr["weighted"],
             label='Average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["weighted"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', "lime", 'magenta'])
    for i, color in zip(range(len(unique)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f"ROC curve of {class_names[i]} (area = {round(roc_auc[i], 3)})")

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize, fontname=fontname)
    plt.ylabel('True Positive Rate', fontsize=fontsize, fontname=fontname)
    plt.title('Receiver operating characteristic', fontsize='14', fontname=fontname)
    plt.legend(loc="lower right", borderaxespad=0.0, frameon=True, edgecolor='k')

    if filename != None:
        fig.savefig(filename, bbox_inches='tight')
    if show_plot:
        plt.show()


def get_roc(targets, preds, class_names, filename=None, show_plot=False):
    """
    Convert predictions and targets to binary values and calculate
    the AUC and ROC. Plot ROC
    """
    targets = check_int(targets) #ensure correct data-type
    preds = check_int(preds)

    unique = np.unique(targets, return_counts=False)
    y_pred = convert_to_binary(preds, unique)
    y_true = convert_to_binary(targets, unique)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(unique)):
        fpr[i], tpr[i], _ = roc_curve(y_pred[:, i], y_true[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    roc_plot(fpr, tpr, roc_auc, unique, class_names, filename=filename, show_plot=show_plot)


def combine_subjects(arr):
    df = pd.DataFrame()
    for n, value in enumerate(arr):
        df[str(n)] = value
    return df

def plot_acc_or_loss(x, y_train, y_valid, y_test, ylabel="", filename=None, show_plot=False):
    """
    PLotting function for accuracy or loss scores.
    """
    plt.style.use('seaborn-whitegrid')
    plt.plot(x, y_train, label='train')
    plt.plot(x, y_valid, label='valid')
    plt.plot(x, y_test, label='test')
    plt.title(f"Training/Validation/Testing {ylabel}", fontsize=14, fontname='Times New Roman')
    plt.xlabel('Epochs', fontsize=14, fontname='Times New Roman')
    plt.ylabel(f"{ylabel}", fontsize=14, fontname='Times New Roman')
    plt.xlim([0, len(x)])
    plt.legend(frameon=True, edgecolor='k', fontsize=14, framealpha=1.0)
    if filename != None:
        plt.savefig(filename, bbox_inches='tight')
    if show_plot:
        plt.show()


def get_mean(results_object, metric="loss"):
    """
    Function for combining either the loss or accuracy scores from each subjects outer-fold results. The resulting
    subject-combined scores are averaged to facilitate mean epoch-by-epoch scores
    :param results_object: python object containing all-subject results
    :param metric: (str) must be "loss" or "acc"
    :return: mean_train: (pandas.DataFrame) mean training scores across subjects
             mean_valid: (pandas.DataFrame) mean validation scores across subjects
             mean_test: (pandas.DataFrame) mean testing scores across subjects
    """

    assert metric == "loss" or metric == "acc", "metric must be loss or acc to use this function"

    combined_train = combine_subjects(getattr(results_object, f"combined_train_{metric}"))
    combined_valid = combine_subjects(getattr(results_object, f"combined_valid_{metric}"))
    combined_test = combine_subjects(getattr(results_object, f"combined_test_{metric}"))

    train_plot = pd.DataFrame(combined_train)
    valid_plot = pd.DataFrame(combined_valid)
    test_plot = pd.DataFrame(combined_test)
    train_plot.fillna(0, inplace=True)
    valid_plot.fillna(0, inplace=True)
    test_plot.fillna(0, inplace=True)
    mean_train = train_plot.mean(axis=1, skipna=True)
    mean_valid = valid_plot.mean(axis=1, skipna=True)
    mean_test = test_plot.mean(axis=1, skipna=True)

    return mean_train, mean_valid, mean_test

def scattered_boxplot(data, xlabel='paradigm', ylabel='metric', title='', filename=None, show_plot=False):
    import matplotlib.style as style
    sns.set(style="white")
    data = check_array(data, ensure_2d=False)

    P.figure(figsize = (6,6))

    bp = P.boxplot(data)

    x = np.random.normal(1, 0.04, size=len(data))
    P.plot(x, data, 'r.', alpha=0.2)
    P.xlabel(f"{xlabel}", fontname='Times New Roman', fontsize=18)
    P.ylabel(f"{ylabel}", fontname='Times New Roman', fontsize=18)
    P.title(title, fontname='Times New Roman', fontsize=20)

    P.tight_layout()
    if filename != None:
        P.savefig(filename, dpi=80)

    if show_plot:
        P.show()

