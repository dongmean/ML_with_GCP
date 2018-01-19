# ----------------------------------------
# Library used for command parsing (maybe)
# ----------------------------------------
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
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import os
import matplotlib.pyplot as plt
import csv



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
	print(data)


	# train Parameters
	seq_length = 10
	data_dim = 52
	hidden_dim = 10
	output_dim = 1
	learning_rate = 0.01
	iterations = 100

	#이쪽으로 csv파일에서 data를 가져오는것만 잘해주면 될듯..! 

	#normalize
	xy = MinMaxScaler(data)
	x = xy
	y = xy[:,[-1]]

	#build a dataset
	data_X = []
	data_Y = []
    
	print("=====data setting started=====")
    
	for i in range(0, len(y) - seq_length):
		_x = x[i:i+seq_length]
	 	_y = y[i+seq_length-1] #last sum_wait_time

	 	#if (i%50 == 0):
			#print(_x,"->",_y)

		data_X.append(_x)
		data_Y.append(_y)

	print("=====data setting finished=====")

	#train/test split
	train_size = int(len(data_Y)*0.7)
	test_size = len(data_Y) - train_size

	train_X, test_X = np.array(data_X[0:train_size]), np.array(data_X[test_size:len(data_X)])
	train_Y, test_Y = np.array(data_Y[0:train_size]), np.array(data_Y[test_size:len(data_X)])

	print("=====train/test splitted=====")

	#input place holders
	X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
	Y = tf.placeholder(tf.float32, [None, 1])

	#build a LSTM network
	cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_dim, state_is_tuple = True, activation = tf.tanh)

	#activation??

	outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)

	Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn = None)

	#cost/loss
	#loss =  tf.reduce_sum(tf.square(Y_pred - Y)) #잔차의 제곱 합으로 loss를 잡은거..
	loss = tf.reduce_sum(tf.multiply(Y, tf.square(Y_pred-Y)))

	#optimizer, train
	train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	# RMSE
	targets = tf.placeholder(tf.float32, [None, 1])
	predictions = tf.placeholder(tf.float32, [None, 1])
	#rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions))) #기존 RMSE
	rmse = tf.sqrt(tf.reduce_mean(tf.multiply(targets, tf.square(targets - predictions))))

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)

		#Training step
		for i in range(iterations):
			_, step_loss = sess.run([train, loss], feed_dict={X : train_X, Y : train_Y})
			if(i%10 ==0):
				print("step : {} , loss: {}\n".format(i, step_loss))

		#Test step
		test_predict = sess.run(Y_pred, feed_dict={X: test_X})
		rmse_val = sess.run(rmse, feed_dict={targets : test_Y, predictions : test_predict})

		print("==========result==========")
		print("RMSE : {}".format(rmse_val))

		#plotting, file 바로 저장
		plt.figure()        
		plt.plot(test_Y)
		plt.plot(test_predict)
		plt.xlabel("Time Period")
		plt.ylabel("sum of wait time")

		plt.savefig('im5_LSTM_result1.png')
		credentials = GoogleCredentials.get_application_default()
		service = discovery.build('storage', 'v1', credentials=credentials)

		filename = 'im5_LSTM_result1.png'
		bucket = 'model1-ods-im5-os-stat-wait'

		body = {'name': 'loss_fixed/im5_LSTM_result1.png'}
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