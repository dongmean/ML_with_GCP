import argparse
import os
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam

# ---------------------------------------------------
# Library used for loading a file from Google Storage
# ---------------------------------------------------
from tensorflow.python.lib.io import file_io

# ---------------------------------------------------
# Library used for uploading a file to Google Storage
# ---------------------------------------------------
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

import tensorflow as tf
import matplotlib as mpl
mpl.use('agg')

import os
import matplotlib.pyplot as plt
import csv

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

def MinMaxScaler(data):
	''' Min Max Normalization
	Parameters, Returns
	----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    '''
	numerator = data - np.min(data, 0)
	denominator = np.max(data, 0) - np.min(data, 0)
    
    # noise term prevents the zero division
	return numerator / (denominator + 1e-7)

def load_series(filename):
	filename = filename[0] #filename : route (--train-files <route>)
	try:
		with file_io.FileIO(filename, mode='r') as csvfile:
			print("===in load_series function, fileIO===")
			csvreader = csv.reader(csvfile)
			data = [row for row in csvreader if len(row) > 0]
			return data
	except IOError:
		return None

def sampling_data(data, leg_size, slide_len):
    
	# This function is for time series data sampling
	# leg_size : Length of splited data
	# window_size : Sliding window size
    
	samples = []
	for pos in range(0,len(data),slide_len):
		sample = np.copy(data[pos:pos+leg_size])
		if len(sample) != leg_size:
			continue
		samples.append(sample)
	return samples

def reconstruct(data, window, clusterer):
	# reconstruct data with centeroid of clusterer

	# data is input data
	# window is window function that make (input data's starting point and ending point) to be zero
	# Clusterer : scikit-learn Cluster model

	window_len = len(window)
	slide_len = int(window_len/2)

	[n, e] = data.shape
	print(e)

	# data spliting with sliding lenght window_len/2 ( some datas are overlapping )
	segments = sampling_data(data, window_len, slide_len)
    
	print("====test_X segments shape in reconstruct====")
	print(np.array(segments).shape)

	reconstructed_data = np.zeros((len(data), e))

	# find nearest centroid among clusters for reconstruction
	for segment_n, segment in enumerate(segments):
		segment *= window

		segment = np.reshape(segment,(1, (window_len*e) ))

		nearest_match_idx = clusterer.predict(segment)[0] #sample 한개 넣었으니 [0]
		nearest_match = np.copy(clusterer.cluster_centers_[nearest_match_idx])

		if(segment_n < 10):
			print(np.array(nearest_match).shape)
			print(window_len*e)

		nearest_match = np.reshape(nearest_match, (window_len, e))

		pos = segment_n*slide_len
        
		if(pos+(window_len) < len(data)):
			reconstructed_data[pos:pos+(window_len)] += nearest_match

	return reconstructed_data


def run_experiment(hparams):
	data = load_series(hparams.train_files)

	print("=====run experiment=====")

	#data가 string의 list라 float형의 np.array로 casting 해준다
	data = np.array(data)
	data = np.delete(data, (0), axis=0)
	data = data.astype(float)
	#print(data)

	#standardization
	#xy = MinMaxScaler(data)
    
	xy = data
	x = xy[:,0:-1]

	#build a dataset
	print("========data building started========")

	data_X = np.array(x)
    
	#normalization
	data_X = normalize(data_X, axis=0, norm='l2')

    
	#train/test split
	print("=====train/test split started=====")
    
	train_size = int(len(data_X)*0.8)

	train_X, test_X = np.array(data_X[0:train_size]), np.array(data_X[train_size:len(data_X)])
	#train_Y, test_Y = np.array(data_Y[0:train_size]), np.array(data_Y[train_size:len(data_X)])

	#hyperparameter
	window_len = 16
	n_clusters = 150

	# window function is sin^2
	window_rads = np.linspace(0,np.pi,window_len)
	window = np.sin(window_rads)**2

	window = np.reshape(window, (-1,1))
	print(window)
    
    
	#sampling
	print("========segmenting started========")
    
	segments = sampling_data(train_X,window_len,2)
    
	print("========segments shape========")
	print(np.array(segments).shape)
    
	windowed_segments = []

	for segment in segments:
		segment *= window
		windowed_segments.append(segment)

	windowed_segments = np.array(windowed_segments)    
	print(windowed_segments.shape)

	[n,w,e] = windowed_segments.shape
	k = w*e
	windowed_segments = np.reshape(windowed_segments, (n, k)); 

	print(windowed_segments.shape)
    
	windowed_segments = windowed_segments.tolist()
    
	#Clustering
	print("========modeling(clustering) started========")

	clusterer = KMeans(n_clusters=n_clusters)
	clusterer.fit(windowed_segments)

	#Reconstructing
	print("========predict(reconstruct) started========")

	reconstruction = reconstruct(test_X,window,clusterer)
    
	#Anomaly 검출
	print("========Checking anomaly started========")

	error = reconstruction - test_X
    
	error = np.sum(error**2, axis=1)
    
	print("========saving graph started========")
	#plotting, file 바로 저장
    
	plt.figure()
	#n_plot_samples = 500
	plt.plot(test_X[:,40],label="event 1 wait time") #파  
	plt.plot(test_X[:,41],label="event 2 wait time") #주
	plt.plot(test_X[:,42],label="event 3 wait time") #초  
	plt.plot(test_X[:,43],label="event 4 wait time") #
	plt.plot(test_X[:,44],label="event 5 wait time") #  
	plt.plot(test_X[:,45],label="event 6 wait time") #
	plt.plot(test_X[:,46],label="event 7 wait time") #  
	plt.plot(test_X[:,47],label="event 8 wait time") #
	plt.plot(test_X[:,48],label="event 9 wait time") #  
	plt.plot(test_X[:,49],label="event 10 wait time") #
	plt.plot(test_X[:,50],label="event 11 wait time") #   
	#plt.plot(reconstruction[0:n_plot_samples][50],label="reconstruction");
	#plt.plot(error,label="error")
    
	plt.xlabel("Time Index")
    
	plt.savefig('STS_Kmeans_6_entire_wait_times.png')
	credentials = GoogleCredentials.get_application_default()
	service = discovery.build('storage', 'v1', credentials=credentials)

	filename = 'STS_Kmeans_6_entire_wait_times.png'
	bucket = 'im5-os-stat-wait-sts-kmeans'

	body = {'name': 'graphs/STS_Kmeans_6_entire_wait_times.png'}
	req = service.objects().insert(bucket=bucket, body=body, media_body=filename)
	resp = req.execute()

	plt.show()
    
if __name__ == '__main__':

	# ---------------------------------------------
	# command parsing from Google ML Engine Example
	# ---------------------------------------------
	parser = argparse.ArgumentParser()
	# Input Arguments
	parser.add_argument(
	  '--train-files',
	  help='GCS or local paths to training data',
	  nargs='+',
	  required=True
	)
	parser.add_argument(
	  '--num-epochs',
	  help="""\
	  Maximum number of training data epochs on which to train.
	  If both --max-steps and --num-epochs are specified,
	  the training job will run for --max-steps or --num-epochs,
	  whichever occurs first. If unspecified will run for --max-steps.\
	  """,
	  type=int,
	)
	parser.add_argument(
	  '--train-batch-size',
	  help='Batch size for training steps',
	  type=int,
	  default=40
	)
	parser.add_argument(
	  '--eval-batch-size',
	  help='Batch size for evaluation steps',
	  type=int,
	  default=40
	)

	# -------------------------------
	# If evaluation file is prepared,
	# change 'required' value
	# -------------------------------
	parser.add_argument(
	  '--eval-files',
	  help='GCS or local paths to evaluation data',
	  nargs='+',
	  required=False
	)
	# Training arguments
	parser.add_argument(
	  '--embedding-size',
	  help='Number of embedding dimensions for categorical columns',
	  default=8,
	  type=int
	)
	parser.add_argument(
	  '--first-layer-size',
	  help='Number of nodes in the first layer of the DNN',
	  default=100,
	  type=int
	)
	parser.add_argument(
	  '--num-layers',
	  help='Number of layers in the DNN',
	  default=4,
	  type=int
	)
	parser.add_argument(
	  '--scale-factor',
	  help='How quickly should the size of the layers in the DNN decay',
	  default=0.7,
	  type=float
	)
	parser.add_argument(
	  '--job-dir',
	  help='GCS location to write checkpoints and export models',
	  required=True
	)

	# Argument to turn on all logging
	parser.add_argument(
	  '--verbosity',
	  choices=[
	      'DEBUG',
	      'ERROR',
	      'FATAL',
	      'INFO',
	      'WARN'
	  ],
	  default='INFO',
	)
	# Experiment arguments
	parser.add_argument(
	  '--train-steps',
	  help="""\
	  Steps to run the training job for. If --num-epochs is not specified,
	  this must be. Otherwise the training job will run indefinitely.\
	  """,
	  type=int
	)
	parser.add_argument(
	  '--eval-steps',
	  help='Number of steps to run evalution for at each checkpoint',
	  default=100,
	  type=int
	)
	parser.add_argument(
	  '--export-format',
	  help='The input format of the exported SavedModel binary',
	  choices=['JSON', 'CSV', 'EXAMPLE'],
	  default='JSON'
	)


	args = parser.parse_args()

	# Set python level verbosity
	tf.logging.set_verbosity(args.verbosity)
	# Set C++ Graph Execution level verbosity
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
	  tf.logging.__dict__[args.verbosity] / 10)

	# Run the training job
	hparams=hparam.HParams(**args.__dict__)
	run_experiment(hparams)