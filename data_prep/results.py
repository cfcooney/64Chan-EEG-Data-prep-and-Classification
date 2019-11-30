from dataframe_utils import results_df, get_col_list, param_scores_df
from utils import load_pickle
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Results():

	direct = 'C:/Users/sb00745777/OneDrive - Ulster University/Study_3/Subject_Data'

	def __init__(self, identifier, paradigm, folds=5):
		self.id = identifier
		self.n_folds = folds
		self.paradigm = paradigm
		self.y_true_list = []
		self.y_true = np.array([])
		self.y_pred_list = []
		self.y_pred = np.array([])
		self.y_probs = None
		#self.class_names = class_names
		self.lossdf = None
		self.accdf = None
		self.cross_entropydf = None
		self.subject_stats_df = None
		self.best_params = None
		self.hyp_param_means = []
		self.outer_fold_accuracies = []  # list of scores - 1 per fold
		self.outer_fold_cross_entropies = []
		self.of_mean = None
		self.of_std = None
		self.accuracy = None
		self.precision = None
		self.f1_score = None
		self.recall = None
		self.precision_list = []
		self.f1_score_list = []
		self.recall_list = []
		self.cm = None
		self.train_loss = None
		self.test_loss = None
		self.valid_loss = None
		self.train_acc = None
		self.test_acc = None
		self.valid_acc = None

	def change_directory(self, direct):
		self.direct = direct

	def concat_y_true(self, y_true_fold):
		"""
		Method for combining all outer-fold ground-truth values.
		:param y_true_fold: array of single-fold true values.
		:return: all outer fold true values in single arrau
		"""
		self.y_true = np.concatenate((self.y_true, np.array(y_true_fold)))

	def concat_y_pred(self, y_pred_fold):
		"""
		Method for combining all outer-fold ground-truth values.
		:param y_pred_fold: array of single-fold true values.
		:return: all outer fold true values in single arrau
		"""
		self.y_pred = np.concatenate((self.y_pred, np.array(y_pred_fold)))

	def append_y_true(self, y_true_fold):
		"""
			Method for combining all outer-fold ground-truth values.
			:param y_true_fold: array of single-fold true values.
			:return: list of outer fold true values. Each element contains one fold
			"""
		self.y_true_list.append((np.array(y_true_fold)))

	def append_y_pred(self, y_pred_fold):
		"""
			Method for combining all outer-fold ground-truth values.
			:param y_pred_fold: array of single-fold true values.
			:return: list of outer fold true values. Each element contains one fold
			"""
		self.y_pred_list.append((np.array(y_pred_fold)))

	def get_acc_loss_df(self, hyp_params, index_name):
		# 2 -- Main Accruacy/loss DataFrame for innerfold
		index = list(n+1 for n in range(self.n_folds*self.n_folds))
		index.append("Mean")
		index.append("Std.")
		columns_list = get_col_list(hyp_params)
		names = list(hyp_params.keys())

		self.lossdf = results_df(index,index_name,columns_list,names)
		self.accdf  = results_df(index,index_name,columns_list,names)
		self.cross_entropydf = results_df(index,index_name,columns_list,names)

	def fill_acc_loss_df(self, inner_fold_accs, inner_fold_loss, inner_fold_CE, save=True):
		"""
		Method for inserting all inner-fold accuracies and losses associated with each hyper-parameter
		combination in a dataframe. Mean and Std. computed. The dataframes can be used to select optimal
		hyper-parameters.
		:param inner_fold_accs: list containing all inner-fold accuracy scores
		:param inner_fold_loss: list containing all inner-fold loss values
		:param save: Boolean
		:return: Dataframes in which each column represents a particular hyper-parameter set.
		"""
		for n, acc in enumerate(inner_fold_accs):
			self.accdf.iloc[n] = acc
		self.accdf.loc["Mean"].iloc[0] = self.accdf.iloc[1:(self.n_folds*self.n_folds)].mean(axis=0).values
		self.accdf.loc["Std."].iloc[0] = self.accdf.iloc[1:(self.n_folds*self.n_folds)].std(axis=0).values

		for n, loss in enumerate(inner_fold_loss):
			self.lossdf.iloc[n] = loss
		self.lossdf.loc["Mean"].iloc[0] = self.lossdf.iloc[1:(self.n_folds*self.n_folds)].mean(axis=0).values
		self.lossdf.loc["Std."].iloc[0] = self.lossdf.iloc[1:(self.n_folds*self.n_folds)].std(axis=0).values

		if save:
			self.accdf.to_excel(f"{self.direct}/S{self.id}/Results/{self.paradigm}_HP_acc.xlsx")
			self.lossdf.to_excel(f"{self.direct}/S{self.id}/Results/{self.paradigm}_HP_loss.xlsx")

		if inner_fold_CE is not None:
			for n, ce in enumerate(inner_fold_CE):
				self.cross_entropydf.iloc[n] = ce
			self.cross_entropydf.loc["Mean"].iloc[0] = self.cross_entropydf.iloc[1:(self.n_folds * self.n_folds)].mean(axis=0).values
			self.cross_entropydf.loc["Std."].iloc[0] = self.cross_entropydf.iloc[1:(self.n_folds * self.n_folds)].std(axis=0).values
			self.cross_entropydf.to_excel(f"{self.direct}/S{self.id}/Results/{self.paradigm}_HP_CE.xlsx")


	def get_best_params(self, selection_method):
		"""
		Method for returning best hyper-parameter combination from inner fold accuracy or loss.
		:param selection_method: str: "accuracy" Or "loss".
		:return: list of optimal hyper-parameters.
		"""
		if selection_method == "accuracy":
			self.best_params = list(self.accdf.columns[self.accdf.loc["Mean"].values.argmax()])
		else:
			self.best_params = list(self.lossdf.columns[self.lossdf.loc["Mean"].values.argmin()])
		best_params = pd.DataFrame(dict(best_params=self.best_params))
		best_params.to_excel(f"{self.direct}/S{self.id}/Results/{self.paradigm}_BestParameters.xlsx")

	def get_hp_means(self, hyp_params, selection_method):
		columns_list = get_col_list(hyp_params)
		for HP in columns_list:
			for value in HP:
				if selection_method == 'accuracy':
					sub_df = self.accdf[[i for i in self.accdf.columns if i[0] == value or i[1] == value or i[2] == value]] #or i[3] == value
					self.hyp_param_means.append(sub_df.loc["Mean"].values.mean())
				else:
					sub_df = self.lossdf[[i for i in self.lossdf.columns if i[0] == value or i[1] == value or i[2] == value]] #or i[3] == value
					self.hyp_param_means.append(sub_df.loc["Mean"].values.mean())

	def set_outer_fold_accuracies(self, outer_fold_accuracies):
		self.outer_fold_accuracies = outer_fold_accuracies
		self.of_mean = np.mean(outer_fold_accuracies)
		self.of_std = np.std(outer_fold_accuracies)

	def get_accuracy(self):
		"""
		Method for calculating accuracy from all true and predicted values.
		:return: accuracy value (%) rounded to 3 decimal places.
		"""
		assert len(self.y_true) == len(self.y_pred), "data must be of equal length"
		self.accuracy = round((accuracy_score(self.y_true, self.y_pred) * 100), 3)

	def get_precision(self):
		assert len(self.y_true) == len(self.y_pred), "data must be of equal length"
		self.precision = round((precision_score(self.y_true, self.y_pred, average="macro") * 100), 3)

	def get_recall(self):
		assert len(self.y_true) == len(self.y_pred), "data must be of equal length"
		self.recall = round((recall_score(self.y_true, self.y_pred, average='macro') * 100), 3)

	def get_f_score(self):
		assert len(self.y_true) == len(self.y_pred), "data must be of equal length"
		self.f1_score = round((f1_score(self.y_true, self.y_pred, average='macro') * 100), 3)

	def precision_recall_f_score(self):
		precision_recall_fscore_support(self.y_true, self.y_pred)

	def confusion_matrix(self):
		self.cm = confusion_matrix(self.y_true, self.y_pred)

	def subject_stats(self):
		"""
		Method for constructing and saving a Pandas Dataframe with Accuracy and
		statistical scores as below:
			fold 1  fold 2    Mean   Std.  Precision  Recall  F1 Score
		01  18.065  16.779  17.422  0.643     16.447  16.447    16.447
		"""

		folds = []
		for i in range(1, self.n_folds+1):
			folds.append(f'fold {i}')

		if np.array(self.outer_fold_accuracies).ndim == 1:
			self.subject_stats_df = pd.DataFrame(index=[self.id], columns=folds)
			self.subject_stats_df.iloc[0] = self.outer_fold_accuracies
			self.subject_stats_df['Subj Mean'] = self.subject_stats_df.mean(axis=1, skipna=True)
			self.subject_stats_df['Subj Std.'] = self.subject_stats_df.std(axis=1, skipna=True)
			self.get_precision()
			self.get_recall()
			self.get_f_score()
			self.subject_stats_df['Precision'] = self.precision
			self.subject_stats_df['Recall'] = self.recall
			self.subject_stats_df['F1 Score'] = self.f1_score
			for n,ce in enumerate(self.outer_fold_cross_entropies):
				self.subject_stats_df[f"CE - fold {n+1}"] = ce
			self.subject_stats_df["CE mean"] = np.mean(self.outer_fold_cross_entropies)
			self.subject_stats_df["CE std."] = np.std(self.outer_fold_cross_entropies)

			handle = f"{self.direct}/S{self.id}/Results/{self.paradigm}_statistics.xlsx"

		else:
			self.subject_stats_df = pd.DataFrame(index=[self.ids], columns=folds)
			for n,score in enumerate(self.outer_fold_accuracies):
				self.subject_stats_df.iloc[n] = score
			self.subject_stats_df['Subj Mean'] = self.subject_stats_df.mean(axis=1, skipna=True)
			self.subject_stats_df['Subj Std.'] = self.subject_stats_df.std(axis=1, skipna=True)
			self.subject_stats_df['Precision'] = self.precision_list
			self.subject_stats_df['Recall'] = self.recall_list
			self.subject_stats_df['F1 Score'] = self.f1_score_list

			# adding cross-entropy values for each fold
			for n,_ in enumerate(folds):
				self.subject_stats_df[f"CE - fold {n+1}"] = ""
			for n,ce_list in enumerate(self.outer_fold_cross_entropies):
				for m,ce in enumerate(ce_list):
					self.subject_stats_df[f"CE - fold {m+1}"].iloc[n] = ce
			self.subject_stats_df["CE mean"] = self.outer_fold_ce_means
			self.subject_stats_df["CE std."] = self.outer_fold_ce_std

			self.subject_stats_df.loc["Mean"] = self.subject_stats_df.iloc[0:len(self.ids)].mean(axis=0).values
			self.subject_stats_df.loc["Std."] = self.subject_stats_df.iloc[0:len(self.ids)].std(axis=0).values

			handle = f"{self.direct}/results/{self.paradigm.replace('EEG_', '')}/combined_stats.xlsx"

		self.subject_stats_df.to_excel(handle)

	def save_result(self):
		filename = f"{self.direct}/S{self.id}/Results/{self.paradigm}_results_object.pickle"
		filehandler = open(filename, 'wb')
		subject = dict(subject=self)
		pickle.dump(subject, filehandler, protocol=pickle.HIGHEST_PROTOCOL)


class CombinedResults(Results):


	def __init__(self, identifier, paradigm, folds, filename, ids):

		super().__init__(identifier, paradigm, folds)

		self.filename = filename
		self.ids = ids
		self.total_cross_val_df = None
		#self.total_fold_accuracies = []
		self.total_best_hps = [] #list of best HPs for each subject
		self.BestParams = None
		self.hp_results_df = None
		self.outer_fold_ce_means = []
		self.outer_fold_ce_std = []
		self.combined_train_loss = []
		self.combined_test_loss = []
		self.combined_valid_loss = []
		self.combined_train_acc = []
		self.combined_test_acc = []
		self.combined_valid_acc = []
		self.HP_acc = pd.DataFrame(columns=self.ids)
		self.HP_loss = pd.DataFrame(columns=self.ids)
		self.HP_ce = pd.DataFrame(columns=self.ids)

	def cross_val_results_df(self, handle):
		assert len(self.outer_fold_accuracies) == len(self.ids), "Number of subjects and results are not equal"
		assert len(self.outer_fold_accuracies[0]) == self.n_folds, "Number of scores and folds are not equal"
		folds = []
		for i in range(1, self.n_folds+1):
			folds.append(f'fold {i}')
			self.total_cross_val_df = pd.DataFrame(index=self.ids, columns=folds)

		for n,score in enumerate(self.outer_fold_accuracies):
			self.total_cross_val_df.iloc[n] = score

		self.total_cross_val_df['Mean'] = self.total_cross_val_df.mean(axis=1,skipna=True)
		self.total_cross_val_df['Std.'] = self.total_cross_val_df.std(axis=1,skipna=True)

		self.total_cross_val_df.to_excel(f'{self.direct}/results/{handle}/cross_val_scores.xlsx')

	def get_subject_results(self):

		for i in self.ids:
			try:
				results_object, _ = load_pickle(f"{self.direct}/S{i}/","Results",f"{self.paradigm}_{self.filename}.pickle")
			except:
				results_object, _ = load_pickle(f"{self.direct}/S{i}/", f"Session_2", f"{self.filename}.pickle")
			else:
				print(f"")

			self.y_true = np.concatenate((self.y_true, results_object['subject'].y_true))
			self.y_pred = np.concatenate((self.y_pred, results_object['subject'].y_pred)) # all true and prediction values

			self.outer_fold_accuracies.append(results_object['subject'].outer_fold_accuracies)
			self.outer_fold_cross_entropies.append(results_object['subject'].outer_fold_cross_entropies)
			self.outer_fold_ce_means.append(np.mean(results_object['subject'].outer_fold_cross_entropies))
			self.outer_fold_ce_std.append(np.std(results_object['subject'].outer_fold_cross_entropies))

			results_object['subject'].get_precision()
			results_object['subject'].get_f_score()
			results_object['subject'].get_recall()
			self.precision_list.append(results_object['subject'].precision)
			self.f1_score_list.append(results_object['subject'].f1_score)
			self.recall_list.append(results_object['subject'].recall)

			self.total_best_hps.append(results_object['subject'].best_params)
			self.hyp_param_means.append(results_object['subject'].hyp_param_means)

			self.combined_train_loss.append(results_object['subject'].train_loss)
			self.combined_test_loss.append(results_object['subject'].test_loss)
			self.combined_valid_loss.append(results_object['subject'].valid_loss)
			self.combined_train_acc.append(results_object['subject'].train_acc)
			self.combined_test_acc.append(results_object['subject'].test_acc)
			self.combined_valid_acc.append(results_object['subject'].valid_acc)

		# Save combined predictions and ground truth values to csv
		np.savetxt(f"{self.direct}/results/{self.paradigm.replace('EEG_', '')}/y_true.csv", [self.y_true],
				   delimiter=',', fmt='%d')
		np.savetxt(f"{self.direct}/results/{self.paradigm.replace('EEG_', '')}/y_pred.csv", [self.y_pred],
				   delimiter=',', fmt='%d')

	def param_scores(self, hyp_params):
		"""
		Saves a Pandas DataFrame as an Excel file which contains average inner-fold accuracy (or loss)
		for each independent hyperparameter value, and for all subjects
		:param hyp_params: dict containing all hyperparameter keys and values.
		:return:
		"""
		paramscores_df = param_scores_df(self.ids, hyp_params)
		for i, j in enumerate(self.hyp_param_means):
			paramscores_df.iloc[i] = j

		paramscores_df.loc["Mean"] = paramscores_df[0:len(self.ids)].mean(axis=0, skipna=True)
		paramscores_df.loc["Std."] = paramscores_df[0:len(self.ids)].std(axis=0, skipna=True)
		paramscores_df.to_excel(f"{self.direct}/results/{self.paradigm.replace('EEG_', '')}/param_scores.xlsx")

	def inter_subject_hps(self, hyp_params, index_name, selection_method):
		index = list(n + 1 for n in range(len(self.ids)))
		index.append("Mean")
		index.append("Std.")
		columns_list = get_col_list(hyp_params)
		names = list(hyp_params.keys())

		self.hp_results_df = results_df(index, index_name, columns_list, names)
		#self.ids = self.ids[:-2]

		combined_hp = []
		for i in self.ids:
			try:
				results_object, _ = load_pickle(f"{self.direct}/S{i}/", f"Results", f"{self.paradigm}_{self.filename}.pickle")
			except:
				results_object, _ = load_pickle(f"{self.direct}/S{i}/", f"Session_2", f"{self.filename}.pickle")
			# else:
			# 	print(f"No results for subject S{i}")

			results_object = results_object['subject']
			acc = results_object.accdf.loc['Mean'].values
			combined_hp.append(acc)

		for i, j in enumerate(combined_hp):
			self.hp_results_df.iloc[i] = j
		self.hp_results_df.loc["Mean"].iloc[0] = self.hp_results_df.iloc[0:len(self.ids)].mean(axis=0, skipna=True)
		self.hp_results_df.loc["Std."].iloc[0] = self.hp_results_df.iloc[0:len(self.ids)].std(axis=0, skipna=True)
		self.hp_results_df.to_excel(f"{self.direct}/results/{self.paradigm.replace('EEG_', '')}/total_hp_scores.xlsx")


		self.BestParams = self.hp_results_df.columns[self.hp_results_df.loc["Mean"].values.argmax()]
		self.BestParams = pd.DataFrame(dict(BestParams=self.BestParams))
		self.BestParams.to_excel(f"{self.direct}/results/{self.paradigm.replace('EEG_', '')}/BestParams.xlsx")

	def get_combined_inner_scores(self):

		for i in self.ids[:-2]:
			try:
				results_object, _ = load_pickle(f"{self.direct}/S{i}/","Results",f"{self.paradigm}_{self.filename}.pickle")
			except:
				results_object, _ = load_pickle(f"{self.direct}/S{i}/", f"Session_2", f"{self.filename}.pickle")
			else:
				print(f"")

			self.HP_acc[i]  = results_object['subject'].accdf.loc['Mean'].apply(lambda x : x * 100).values.ravel()
			self.HP_loss[i] = results_object['subject'].lossdf.loc['Mean'].values.ravel()
			self.HP_ce[i]   = results_object['subject'].cross_entropydf.loc['Mean'].values.ravel()

		self.HP_acc.fillna(0, inplace=True) # zero-filling -- mean-filling may be a better option
		self.HP_loss.fillna(0, inplace=True)
		self.HP_ce.fillna(0, inplace=True)
		self.HP_acc['Mean'] = self.HP_acc.mean(axis=1, skipna=True)
		self.HP_loss['Mean'] = self.HP_loss.mean(axis=1, skipna=True)
		self.HP_ce['Mean'] = self.HP_ce.mean(axis=1, skipna=True)

	def save_combined_result(self):
		filename = f"{self.direct}/results/{self.paradigm.replace('EEG_', '')}/combineResults.pickle"
		filehandler = open(filename, 'wb')
		subject = dict(subject=self)
		pickle.dump(subject, filehandler, protocol=pickle.HIGHEST_PROTOCOL)





