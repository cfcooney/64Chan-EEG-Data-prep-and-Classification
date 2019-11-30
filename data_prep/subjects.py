from utils import load_pickle, timer
from scipy.signal import decimate as dec
from tensorflow.keras.utils import normalize
from scipy.signal import butter, lfilter
import numpy as np
import pickle
import warnings 
warnings.filterwarnings('ignore', category=FutureWarning)

class Subject(object):

	direct = 'C:/Users/cfcoo/OneDrive - Ulster University/Study_3/Subject_Data'

	def __init__(self, id, epoch):
		self.id = id
		self.n_chans = 64
		self.epoch = epoch
		self.channels_validated = False
		self.trials_validated = False
		self.description = None
		self.data_loaded = False
		self.data = None
		self.data3D = None
		self.labels = None
		self.sfreq = None
		self.downsample_rate = 2
		self.downsampled = False
		self.normalized = False
		self.filtered = False
		self.get_3D = None
		self.lowcut = 2
		self.highcut = 40

	def set_description(self, description):
		self.description = description 

	def get_description(self):
		return self.description

	def change_directory(self, new_direct):
		self.direct = new_direct

	def set_channel_validation(self, validated):
		assert type(validated) == bool
		self.channels_validated = validated 

	def get_channel_validation(self):
		return self.channels_validated

	def set_trial_validation(self, validated):
		assert type(validated) == bool
		self.trials_validated = validated 

	def get_trial_validation(self):
		return self.trials_validated

	def eeg_to_3d(self):
	    """
	    function to return a 3D EEG data format from a 2D input.
	    Parameters:
	      data: 2D np.array of EEG
	      epoch_size: number of samples per trial, int
	      n_events: number of trials, int
	      n_chan: number of channels, int
	        
	    Output:
	      np.array of shape n_events * n_chans * n_samples
	    """
	    idx, a, x = ([] for i in range(3))
	    [idx.append(i) for i in range(0,self.data.shape[1],self.epoch)]
	    for j in self.data:
	        [a.append([j[idx[k]:idx[k]+self.epoch]]) for k in range(len(idx))]
	        
	    return np.reshape(np.array(a),(self.labels.shape[0],self.n_chans,self.epoch))


	def bandpass(self, lowcut, highcut, order):
		"""
		Filter first, then downsample.
		"""
		assert self.downsampled == False, "Signals must be filtered before downsampling"
		self.lowcut = lowcut
		self.highcut = highcut
		nyq = 0.5 * self.sfreq
		low = self.lowcut / nyq
		high = self.highcut / nyq
		b, a = butter(order, [low, high], btype='band')

		self.data = lfilter(b, a, self.data)
		self.data3D = lfilter(b, a, self.data3D)

		self.filtered = True

	
	def down_and_normal(self, downsample_rate, norm):
		"""
		Method for downsampling the data, normalizing 
		and improving numerical stability.
		"""
		assert self.data_loaded == True, "Data must be loaded to access this method!" #change to correct shape, format etc.

		self.downsample_rate = downsample_rate
		self.data = dec(self.data, downsample_rate) #downsampling
		
		fnc = lambda a: a * 1e6 # improves numerical stability
		self.data = fnc(self.data)
		if norm:
			self.data = normalize(self.data)
		
		if self.get_3D:
			self.data3D = dec(self.data3D, self.downsample_rate) 
			self.data3D = fnc(self.data3D) 
			if norm:
				self.data3D = normalize(self.data3D)
		
		self.normalized  = True
		self.downsampled = True
		self.epoch = self.data3D.shape[2]

	@timer
	def load_data(self, session, filename, get_3D):
		"""
		Load previously-validated EEG data and labels in the form of an MNE Raw Array.

		Returns: n_chans * n_samples Numpy array contianing EEG data.
				 data3D: n_trials * n_chans * n_samples reshaped EEG data.
		         Numpy array containing labels for all trials.
		         sfreq: sampling frequency of EEG.
		"""
		self.get_3D=get_3D
		

		dataPickle, _ = load_pickle(f"{self.direct}/S{self.id}/", f"Session_{session}/EpochedArrays-postICA", f"{filename}.pickle")

		self.data = dataPickle['EEG'].get_data()
		self.n_chans = self.data.shape[0]
		self.labels = dataPickle['labels']
		
		# import random
		# self.labels = np.array([random.randint(1,4) for i in range(200)]) #np.ma.masked_equal(labels,0).compressed()
		
		self.sfreq = dataPickle['EEG'].info['sfreq']
	
		self.data_loaded = True

		if self.get_3D:
			self.data3D = self.eeg_to_3d()

	def save_subject(self, session, filename):
		filename = f"{self.direct}/S{self.id}/Session_{session}/{filename}.pickle"
		filehandler = open(filename, 'wb')
		subject = dict(subject=self)
		pickle.dump(subject, filehandler, protocol=pickle.HIGHEST_PROTOCOL)

	def get_details(self):
		print(f"Subject: {self.id}")
		print("-"*15)
		print(self.description)
		print("-"*15)
		print(f"Data loaded: {self.data_loaded}")
		if self.data_loaded:
			print(f"Data shape: {self.data.shape}")
			if self.get_3D:
				print(f"Data reshaped: {self.data3D.shape}")
			print(f"Labels shape: {self.labels.shape[0]}")
			print(f"Number of  valid channels: {self.n_chans}")
			print(f"Sampling Frequency: {self.sfreq} Hz")
			print(f"Data downsampled: {self.downsampled}")
			if self.downsampled:
				print(f"Downsample Rate: {self.downsample_rate}")
				print(f"Data normalized: {self.normalized}")
			if self.filtered:
				print(f"Data bandpass filtered between {self.lowcut} and {self.highcut} Hz")