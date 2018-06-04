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

# Gaussian MLP as encoder
def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob):
	with tf.variable_scope("gaussian_MLP_encoder"):
		# initializers
		w_init = tf.contrib.layers.variance_scaling_initializer()
		b_init = tf.constant_initializer(0.)

		#print("========")        
		#print(x.get_shape()[1],n_hidden)
		# 1st hidden layer
		w0 = tf.get_variable('w0', [x.shape[1], n_hidden], initializer=w_init)  #shape?
		b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
		h0 = tf.matmul(x, w0) + b0
		h0 = tf.nn.elu(h0)
		#h0 = tf.nn.dropout(h0, keep_prob)

		# 2nd hidden layer
		w1 = tf.get_variable('w1', [h0.shape[1], n_hidden], initializer=w_init)
		b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
		h1 = tf.matmul(h0, w1) + b1
		h1 = tf.nn.tanh(h1)
		#h1 = tf.nn.dropout(h1, keep_prob)

		# output layer
		# borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
		wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2], initializer=w_init)
		bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
		gaussian_params = tf.matmul(h1, wo) + bo

		# The mean parameter is unconstrained
		mean = gaussian_params[:, :n_output]
		# The standard deviation must be positive. Parametrize with a softplus and
		# add a small epsilon for numerical stability
		stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])

	return mean, stddev

# Bernoulli MLP as decoder
def bernoulli_MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):

	with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
		# initializers
		w_init = tf.contrib.layers.variance_scaling_initializer()
		b_init = tf.constant_initializer(0.)

		# 1st hidden layer
		w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
		b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
		h0 = tf.matmul(z, w0) + b0
		h0 = tf.nn.tanh(h0)
		#h0 = tf.nn.dropout(h0, keep_prob)

		# 2nd hidden layer
		w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
		b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
		h1 = tf.matmul(h0, w1) + b1
		h1 = tf.nn.elu(h1)
		#h1 = tf.nn.dropout(h1, keep_prob)

		# output layer-mean
		wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
		bo = tf.get_variable('bo', [n_output], initializer=b_init)
		y = tf.sigmoid(tf.matmul(h1, wo) + bo)

	return y

# autoencoder
def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob):

	print("========in autoencoder========")
	# encoding
	mu, sigma = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob)

	#print(mu)
	# sampling by re-parameterization technique
	z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

	# decoding
	y = bernoulli_MLP_decoder(z, n_hidden, dim_img, keep_prob)
	y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

	# loss
	marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
	KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

	marginal_likelihood = tf.reduce_mean(marginal_likelihood)
	KL_divergence = tf.reduce_mean(KL_divergence)

	ELBO = marginal_likelihood - KL_divergence

	loss = -ELBO

	#print(loss)
	return y, z, loss, -marginal_likelihood, KL_divergence

def run_experiment(hparams):
	data = load_series(hparams.train_files)

	print("=====run experiment=====")

	#data가 string의 list라 float형의 np.array로 casting 해준다
	data = np.array(data)
	data = np.delete(data, (0), axis=0)
	data = data.astype(float)
	#print(data)

	#standardization
	xy = MinMaxScaler(data)
	x = xy[:,0:-1]

	#hyperparameter
	n_hidden = 60
	seq_length = 10
	dim_train = x.shape[1] #51
	dim_input = dim_train*seq_length #지금은 510
	dim_z = 20
	n_epochs = 1
	batch_size = 100
	learn_rate = 0.00002

	#build a dataset
	print("========data building started========")

	data_X = []

	for i in range(0, len(x) - seq_length):
		_x = x[i:i+seq_length]
		_x = np.reshape(_x, -1) #일렬이 되었을까..?
		data_X.append(_x)

	#train/test split
	print("=====train/test split started=====")
    
	train_size = int(len(data_X)*0.8)

	train_X, test_X = np.array(data_X[0:train_size]), np.array(data_X[train_size:len(data_X)])

	# input placeholders
	# In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
	x_hat = tf.placeholder(tf.float32, shape=[None, dim_input], name='input_data_X')
	x = tf.placeholder(tf.float32, shape=[None, dim_input], name='target_data_X')
    
	# dropout
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	# latent_variable placeholder
	#z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable') #아직안씀

	print("=====modeling started=====")
	# network architecture
	y, z, loss, neg_marginal_likelihood, KL_divergence = autoencoder(x_hat, x, dim_input, dim_z, n_hidden, keep_prob)

	# optimization
	train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

	print("=====training started=====")
	# train
	n_samples = train_X.shape[1]
	print("n_samples : {}".format(n_samples))
	total_batch = int(n_samples / batch_size)
	min_tot_loss = 1e99
    
	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

		for epoch in range(n_epochs):
          
			_, tot_loss, loss_likelihood, loss_divergence = sess.run(
				(train_op, loss, neg_marginal_likelihood, KL_divergence),
				feed_dict={x_hat: train_X, x: train_X, keep_prob : 0.9})

			# print cost every epoch
			print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (epoch, tot_loss, loss_likelihood, loss_divergence))

			# if minimum loss is updated or final epoch, print loss
			if min_tot_loss > tot_loss or epoch+1 == n_epochs:
				print("========testing started========")
				min_tot_loss = tot_loss

				print("epoch : {}, min_tot_loss : {}".format(epoch, min_tot_loss))
				# Decode for test_X
				_y = sess.run(y, feed_dict={x_hat: test_X, keep_prob : 1})
				print("_y : {}".format(_y))
    
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
	ax11.plot(_y[:,50])
	ax12.plot(test_X[:,49])
	ax12.plot(_y[:,49])
	ax13.plot(test_X[:,48])
	ax13.plot(_y[:,48])
	ax21.plot(test_X[:,47])
	ax21.plot(_y[:,47])
	ax22.plot(test_X[:,39])
	ax22.plot(_y[:,39])
	ax23.plot(test_X[:,38])
	ax23.plot(_y[:,38])
	ax31.plot(test_X[:,37])
	ax31.plot(_y[:,37])
	ax32.plot(test_X[:,36])
	ax32.plot(_y[:,36])
	ax33.plot(test_X[:,5])
	ax33.plot(_y[:,5])
    
	plt.savefig('VAE_6_epoch_500.png')
	credentials = GoogleCredentials.get_application_default()
	service = discovery.build('storage', 'v1', credentials=credentials)

	filename = 'VAE_6_epoch_500.png'
	bucket = 'im5-os-stat-wait-vae'

	body = {'name': 'graphs/VAE_6_epoch_500.png'}
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