"""
Name: Ciaran Cooney
Date: 02/09/2019
Description: Main file for extracting results from multiple subjects and combining them into
overall results. A Results object is created for each paradigm and scores related to each
saved as Excel files to a local drive.
"""
from results import Results, CombinedResults
from utils import timer

hyp_params = dict(activation=["elu","relu"],
                  lr=[0.001],
                  epochs=[1,2])

@timer
def combineResults(direct, subjects, paradigm, filename, n_folds, identifier="all_subjects"):
    total_results = CombinedResults(identifier, paradigm, n_folds, filename, subjects) # results object for all subjects
    total_results.change_directory(direct)

    total_results.get_subject_results() # load all subject results for this paradigm

    total_results.subject_stats()

    total_results.inter_subject_hps(hyp_params, "Subject", "accuracy")

    total_results.param_scores(hyp_params) # saves dataframe with average scores per parameter

    total_results.confusion_matrix()

    total_results.get_combined_inner_scores()
    # save the object in folder
    total_results.save_combined_result()

if __name__ == '__main__':

    direct = 'C:/Users/sb00745777/OneDrive - Ulster University/Study_3/Subject_Data'
    subjects = ["01","02"]
    paradigms = ["EEG_semantics_text"]
    filename = "results_object"
    n_folds = 2

    for paradigm in paradigms:
        handle = paradigm.replace('EEG_', '')

        combineResults(direct, subjects, paradigm, filename, n_folds)
