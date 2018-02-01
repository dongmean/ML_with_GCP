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
import GPy

from sklearn.preprocessing import normalize

GPy.plotting.change_plotting_library("matplotlib")

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
	x = MinMaxScaler(x) #시간을 0~1사이의 index로 수정
    
	y = xy[:,-1]

	#build a dataset
	print("========data building started========")

	data_X = []
	data_Y = []
    
	size = int(len(y)*0.001)#일단 체크용으로 1/10 정도의 데이터로만/ 다시 전체 데이터로 만듬
    
	for i in range(0, size): #4000번째 index 부터 시작
		_x = x[i]
		_y = y[i] 

		data_X.append(_x)
		data_Y.append(_y)
    
	data_Y = np.reshape(data_Y, (-1, 1))
	#print(data_X)
	data_X = np.array(data_X)
	data_Y = np.array(data_Y)
    
	#normalization, l1 norm (anomaly에 대해 크게 반응하지 않으려고..?)
	#data_X = normalize(data_X, axis=0, norm='l1')
	#data_Y = normalize(data_Y, axis=0, norm='l1')
    
	print(data_Y.ndim)
	print(data_Y)

	print("=====train/test split started=====")
	#train/test split
	train_size = int(len(data_Y)*0.8)
	test_size = len(data_Y) - train_size

	train_X, test_X = np.array(data_X[0:train_size]), np.array(data_X[train_size:len(data_X)])
	train_Y, test_Y = np.array(data_Y[0:train_size]), np.array(data_Y[train_size:len(data_X)])
    
    
	#전체 데이터로 모델을 만든 후, 그 데이터가 모델의 CI를 넘는지 체크 해보면 됨

	#hyperparameter
	input_dim = 1
	variance = 1
	lengthscale = 0.2

	#kernel
	kernel = GPy.kern.RBF(input_dim, variance = variance, lengthscale = lengthscale)


	#modeling
	print("========modeling started========")

	model = GPy.models.GPRegression(data_X, data_Y, kernel)
	model.optimize(messages = True)
	print(model)


	#predict
	print("========predicting started========")

	Y_pred, Var_pred = model.predict(data_X)
	print("========Y_pred========")
	print(Y_pred)
	print("========Var_pred========")
	print(np.sqrt(Var_pred)) ###
    
	#CI를 이용한 anomaly counting
	print("========counting anomaly started========")
	total_anomaly = 0
	anomaly_indexes = []
	anomalys = []
    
	test_size = len(data_Y)
	print("test_size : {}".format(test_size))
	for i in range(test_size):
		if (Y_pred[i]-1.96*np.sqrt(Var_pred[i]) > data_Y[i] or Y_pred[i]+1.96*np.sqrt(Var_pred[i]) < data_Y[i]):
			total_anomaly +=1
			anomaly_indexes.append(i)
			anomalys.append(data_Y[i])
    
	print("total anomaly : {}".format(total_anomaly))

	print("anomaly_indexes")
	print(anomaly_indexes)

	print("========saving graph started========")
	#plotting, file 바로 저장
    
	plt.figure();
	plt.plot(data_Y, '.'); #파란색
	plt.plot(Y_pred, '.'); #주황색
    
	plt.plot(np.add(Y_pred, 1.96*np.sqrt(Var_pred)))
	plt.plot(np.add(Y_pred, -1.96*np.sqrt(Var_pred)))
    
	plt.xlabel("Time Index")
	plt.ylabel("sum of wait time")
    
	plt.savefig('GP_RBF_only_2_entire_data.png')
	credentials = GoogleCredentials.get_application_default()
	service = discovery.build('storage', 'v1', credentials=credentials)

	filename = 'GP_RBF_only_2_entire_data.png'
	bucket = 'stock-practice-gp'

	body = {'name': 'figures/GP_RBF_only_2_entire_data.png'}
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