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

# -----------------------------------------------------------------
# With use('agg'), import error does not occur
# I used use('TkAgg') to settle error related to my own PC (macOS),
# so if you do not have any problem just ignore it.
# -----------------------------------------------------------------
#import matplotlib as mpl
#mpl.use('agg')
# mpl.use('TkAgg')
#import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import csv


sess= tf.Session()

def load_series(filename ,series_dix=1):
	filename = filename[0]
	try:
		# -------------------------------------------------------
		# 'filename' will be the route from --train-files <route>
		# -------------------------------------------------------
		with file_io.FileIO(filename, mode='r') as csvfile:
			csvreader = csv.reader(csvfile)
			data = [float(row[series_dix]) for row in csvreader if len(row) > 0]
			differencing_data = np.zeros([len(data)-1])

			for i in range(len(data)-3):
				differencing_data[i+1]= data[i]-data[i+1]
			normalized_data = (differencing_data - np.mean(differencing_data)) / np.std(differencing_data)
		return normalized_data, np.mean(differencing_data), np.std(differencing_data)
	except IOError:
		return None


def split_data(data, percent_train=0.60, percent_validation=0.20):
	num_rows = len(data)
	train_data, test_data, validation_data = [], [], []
	for idx, row in enumerate(data):
		if idx < num_rows * percent_train:
			train_data.append(row)
		elif num_rows * percent_train < idx and idx <  num_rows * (percent_train + percent_validation):
			validation_data.append(row)
		else:
			test_data.append(row)
	return train_data, validation_data, test_data

def de_differencing(train_x, predictions, actual, mean, std):
	train_de_x=[]
	predictions_de=[]
	actual_de=[]
	train_de_x.append(1267000)
	train_x = np.multiply(train_x,std)+mean
	predictions = np.multiply(predictions,std)+mean
	actual = np.multiply(actual,std)+mean
	for i in range(len(train_x)):
		train_de_x.append(train_de_x[i]-train_x[i])

	predictions_de.append(train_de_x[-1])
	for i in range(len(predictions)):
		predictions_de.append(predictions_de[i]-predictions[i])

	actual_de.append(train_de_x[-1])
	for i in range(len(actual)):
		actual_de.append(actual_de[i]-actual[i])

	return train_de_x, predictions_de, actual_de


# def plot_results(train_x, predictions, actual):
# 	plt.figure()
# 	num_train = len(train_x)
# 	plt.plot(list(range(num_train)), train_x, color='b', label='training data')
# 	plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
# 	plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
# 	plt.legend()

	# ------------------------------------------------------
	# Save the result figure and upload it to Googld Storage
	# Use your own file name and bucket name
	# ------------------------------------------------------
	plt.savefig('result.png')
	credentials = GoogleCredentials.get_application_default()
	service = discovery.build('storage', 'v1', credentials=credentials)

	filename = 'result.png'
	bucket = 'choichoi'

	body = {'name': 'result.png'}
	req = service.objects().insert(bucket=bucket, body=body, media_body=filename)
	resp = req.execute()
	
	plt.show()

class SeriesPredictor:

	def __init__(self, hparams, input_dim, seq_size, hidden_dim=100):
		self.input_dim = input_dim
		self.seq_size = seq_size
		self.hidden_dim = hidden_dim
		self.hparams = hparams

#		self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
		self.W_out = tf.get_variable('W_out',initializer = tf.random_normal([hidden_dim, 1]))
		self.b_out = tf.get_variable('b_out',initializer = tf.random_normal([1]))
		self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
		self.y = tf.placeholder(tf.float32, [None, seq_size])

		self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
		self.train_op = tf.train.AdamOptimizer(0.004).minimize(self.cost)
		self.saver = tf.train.Saver()

	def model(self):
		cell = rnn.GRUCell(self.hidden_dim, reuse=tf.get_variable_scope().reuse)
		outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
		print(outputs)    
		num_examples = tf.shape(self.x)[0]
		W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
		out = tf.matmul(outputs, W_repeated) + self.b_out
		out = tf.squeeze(out)
		#print(out)
		return out
	

	def train(self, train_x, train_y, validation_x, validation_y):
		tf.get_variable_scope().reuse_variables()
		sess.run(tf.global_variables_initializer())
		max_patience = 2
		patience = max_patience
		min_validation_err = float('inf')
		step = 0
		while patience > 0:
			_, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
			if step % 100 == 0:
				validation_err = sess.run(self.cost, feed_dict={self.x: validation_x, self.y: validation_y})
				print('step: {}\t\ttrain err: {}\t\tvalidation err: {}'.format(step, train_err, validation_err))
				if validation_err < min_validation_err:
					min_validation_err = validation_err
					patience = max_patience
				else:
					patience -= 1
			step += 1
		save_path = self.saver.save(sess, self.hparams.job_dir + 'model.ckpt')
		print("Model saved to {}".format(save_path))
        
	def test(self, sess, test_x):
		tf.get_variable_scope().reuse_variables()
		self.saver.restore(sess, self.hparams.job_dir + "model.ckpt")
		output = sess.run(self.model(), feed_dict={self.x: test_x})

		return output

def run_experiment(hparams):
	# --------------------------------------------------------------------------------
	# I think hparams is a class and contains hyperparameters and other configurations
	# hparams.train_files: route to train file
	# hparams.job_dir: output path
	# ...
	# --------------------------------------------------------------------------------
	seq_size = 20
	predictor = SeriesPredictor(hparams, input_dim=1, seq_size=seq_size, hidden_dim = 10)

	data, mean, std = load_series(hparams.train_files)
	train_data, validation_data, actual_vals = split_data(data)
	train_x, train_y = [], []


	for i in range(len(train_data) - seq_size - 1):
	    train_x.append(np.expand_dims(train_data[i:i+seq_size], axis=1).tolist())
	    train_y.append(train_data[i+1:i+seq_size+1])

	validation_x, validation_y = [], []
	for i in range(len(validation_data) - seq_size - 1):
	    validation_x.append(np.expand_dims(validation_data[i:i+seq_size], axis=1).tolist())
	    validation_y.append(validation_data[i+1:i+seq_size+1])
	    
	test_x, test_y = [], []
	for i in range(len(actual_vals) - seq_size - 1):
	    test_x.append(np.expand_dims(actual_vals[i:i+seq_size], axis=1).tolist())
	    test_y.append(actual_vals[i+1:i+seq_size+1])


	predictor.train(train_x, train_y, test_x, test_y)


	with tf.Session() as sess:
		prev_seq = train_x[-1]
		predicted_vals = []
		for i in range(49):
			next_seq = predictor.test(sess, [prev_seq])
			predicted_vals.append(next_seq[-1])
			prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

	train_de_data, predicted_de_vals, actual_de_vals=de_differencing(train_data, predicted_vals, validation_data, mean, std)
	#plot_results(train_de_data, predicted_de_vals, actual_de_vals)   


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


