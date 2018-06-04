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

	#standardization, scale to [-1,1]
	xy = MinMaxScaler(data)
	x = xy[:,0:-1]
	x = (x*2)-1

	# Scale to [-1,1]??
    
	#hyperparameter
	seq_length = 8

	n_hidden_1 = 200
	n_latent = 100
	n_hidden_2 = 200
    
	n_epochs = 5000
	batch_size = 4000
	learn_rate = 0.005
    
	lossname = 'rmse'

	#build a dataset
	print("========data building started========")

	data_X = []

	for i in range(0, len(x) - seq_length):
		_x = x[i:i+seq_length]
		_x = np.reshape(_x, -1) #일렬이 되었을까..? ㅇㅇ!
		data_X.append(_x)

	#train/test split
	print("=====train/test split started=====")
    
	train_size = int(len(data_X)*0.8)

	train_X, test_X = np.array(data_X[0:train_size]), np.array(data_X[train_size:len(data_X)])
	train_Y= train_X
    
	#print(train_X.shape)
	#print(test_X.shape)

	n_samp, n_input = train_X.shape

	print("=====modeling started=====")
    
	x = tf.placeholder("float", [None, n_input])
    
	# Weights and biases to hidden layer
	Wh = tf.Variable(tf.random_uniform((n_input, n_hidden_1), -1.0 / np.sqrt(n_input), 1.0 / np.sqrt(n_input)))
	bh = tf.Variable(tf.zeros([n_hidden_1]))
	h = tf.nn.tanh(tf.matmul(x,Wh) + bh)

	W_latent = tf.Variable(tf.random_uniform((n_hidden_1, n_latent), -1.0 / np.sqrt(n_hidden_1), 1.0 / np.sqrt(n_hidden_1)))
	b_latent = tf.Variable(tf.zeros([n_latent]))
	h_latent = tf.nn.tanh(tf.matmul(h,W_latent) + b_latent)

	Wh_2 = tf.Variable(tf.random_uniform((n_latent, n_hidden_2), -1.0 / np.sqrt(n_latent), 1.0 / np.sqrt(n_latent)))
	bh_2 = tf.Variable(tf.zeros([n_hidden_2]))
	h_2 = tf.nn.tanh(tf.matmul(h_latent,Wh_2) + bh_2)
    
    # Weights and biases to output layer
	Wo = tf.Variable(tf.random_uniform((n_hidden_2, n_input), -1.0 / np.sqrt(n_hidden_2), 1.0 / np.sqrt(n_hidden_2)))
    #Wo = tf.transpose(Wh_2) # tied weights
	bo = tf.Variable(tf.zeros([n_input]))
	output = tf.nn.tanh(tf.matmul(h_2,Wo) + bo)
    
    # Objective functions
	y = tf.placeholder("float", [None,n_input])
    
	if lossname == 'rmse':
		loss = tf.reduce_mean(tf.square(tf.subtract(y, output)))
	elif lossname == 'cross-entropy':
		loss = -tf.reduce_mean(y * tf.log(output))
    
	# optimization
	train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

	print("=====training started=====")
    
	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		for epoch in range(n_epochs):
          
			sample = np.random.randint(n_samp, size=batch_size)
			batch_xs = train_X[sample][:]
			batch_ys = train_Y[sample][:]
			sess.run(train_op, feed_dict={x: batch_xs, y:batch_ys})
            
			# print loss
			if epoch % 100 == 0:
				print (i, sess.run(loss, feed_dict={x: batch_xs, y:batch_ys}))

		#obtain pred, print loss of test_X
		pred = sess.run( output , feed_dict={x: test_X})
		test_loss = sess.run( loss , feed_dict={x: test_X, y: test_X})
		print("test_loss : {}".format(test_loss))
    
	print("========saving graph started========")
    
    
	#plotting, file 바로 저장

	plt.figure()
	ax11=plt.subplot(331)
	ax12=plt.subplot(332)
	ax13=plt.subplot(333)
	ax21=plt.subplot(334)    
	ax22=plt.subplot(335)
	ax23=plt.subplot(336)
	ax31=plt.subplot(337)
	ax32=plt.subplot(338) 
	ax33=plt.subplot(339)

	ax11.plot(test_X[:,50])
	ax11.plot(pred[:,50])
	ax12.plot(test_X[:,49])
	ax12.plot(pred[:,49])
	ax13.plot(test_X[:,48])
	ax13.plot(pred[:,48])
	ax21.plot(test_X[:,47])
	ax21.plot(pred[:,47])
	ax22.plot(test_X[:,39])
	ax22.plot(pred[:,39])
	ax23.plot(test_X[:,38])
	ax23.plot(pred[:,38])
	ax31.plot(test_X[:,37])
	ax31.plot(pred[:,37])
	ax32.plot(test_X[:,36])
	ax32.plot(pred[:,36])
	ax33.plot(test_X[:,5])
	ax33.plot(pred[:,5])
    
	plt.savefig('AE_6_learningrate_005.png')
	credentials = GoogleCredentials.get_application_default()
	service = discovery.build('storage', 'v1', credentials=credentials)

	filename = 'AE_6_learningrate_005.png'
	bucket = 'adam-models'

	body = {'name': 'im5_os_stat_wait/AE/graphs/AE_6_learningrate_005.png'}
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