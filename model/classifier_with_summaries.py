# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# yapf: disable
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
FLAGS = None

#################################
####### Data Preparation ########
#################################
def data_prepare():
	data_file = "../data/dataset/close_2016-2017.csv"
	#data_file = "../data/dataset/close_2012-2017.csv"
	dataset = pd.read_csv(data_file,index_col=0, sep=',', usecols=[0,1], skiprows=1, names=['date','close'],parse_dates=True)
	print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , " ------ Complete Data Reading")

	# calculate momentum: Mt = (CLOSE(t) -CLOSE(t-1))/CLOSE(t)
	dataset['mmt'] = 0.0
	for i in range(1, len(dataset)):
		dataset['mmt'][i] = (dataset['close'][i] - dataset['close'][i-1]) / dataset['close'][i]
	pd.set_option('mode.chained_assignment',None)
	# classify by counts
	dataset['label'] = 0
	mmt_series = dataset['mmt']
	for i in range(len(dataset)):
		mmt = mmt_series[i]
		if mmt < -0.02:
		    dataset['label'][i] = 0
		elif mmt < -0.005:
		    dataset['label'][i] = 1
		elif mmt < 0.005:
		    dataset['label'][i] = 2
		elif mmt < 0.02:
		    dataset['label'][i] = 3
		else:
		    dataset['label'][i] = 4
	print("---- Label Distribution Check --------")
	print("Total: ",len(dataset['label']))
	print(dataset['label'].value_counts().sort_index())
	print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , " ------ Complete Data Preparation")

"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
def variable_summaries(var):	
    with tf.name_scope('summaries'):
    	mean = tf.reduce_mean(var)
      	tf.summary.scalar('mean', mean)
      	with tf.name_scope('stddev'):
        	stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      	tf.summary.scalar('stddev', stddev)
      	tf.summary.scalar('max', tf.reduce_max(var))
      	tf.summary.scalar('min', tf.reduce_min(var))
      	tf.summary.histogram('histogram', var)

''' time recorder '''
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


############################
###### contruction phase ###
############################
def train():
	tf.reset_default_graph()

	# how long will a span cover, e.g. 20 days (4 tradable weeks)
	TIME_SPAN = 10
	TRAIN_RATIO = 0.9#0.8
	BATCH_SIZE = 3 
	n_neurons = 150 #250
	n_steps = TIME_SPAN
	n_input = 1
	n_output = 5 #5 class
	n_layers = 3 #5#10 #5 #3
	learning_rate = 0.001#0.0005 # # 0.02 # 0.005

	with tf.name_scope('input'):
		X = tf.placeholder(tf.float32, [None, n_steps, n_input], name='X-input')
		y = tf.placeholder(tf.int32, [None], name='y-input') # one sequence one label

	with tf.name_scope('lstm_layers'):
		with tf.variable_scope("lstm", initializer=tf.contrib.layers.variance_scaling_initializer()):
			# define network, 1 layer now
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.relu)
			#cells = tf.contrib.rnn.MultiRNNCell([lstm_cell]*n_layers,state_is_tuple=False)
			cells = tf.contrib.rnn.MultiRNNCell([lstm_cell]*n_layers)
			#cells = lstm_cell
			rnn_outputs, states = tf.nn.dynamic_rnn(cells, X, dtype=tf.float32)

	# Add Dropout
	#is_training = False 
	#keep_prob = 0.75
	#if is_training:
	#    lstm_cell = tf.contrib.rnn.DropoutWrapper(cells, input_keep_prob=keep_prob)

	print("Shape of states before concating BIAS: ", states)
	print("STATES: ", states)

	tf.summary.histogram('lstm_outputs',rnn_outputs)
	tf.summary.histogram('lstm_states', states)

	states = states[0] #only need cell's states, omit hidden state
	print("STATES: ", states)

	tf.summary.histogram('lstm_cell_states', states)
	states = tf.concat(axis=1, values=states) #sum up all neuron's result at final step
	tf.summary.histogram('lstm_cell_states_plus_bias', states)

	with tf.name_scope('fully_connected_layer'):
		fc_layer = fully_connected(states, n_output, activation_fn=None)

	print("Shape of Outputs: ",rnn_outputs, "shape of states: ", states)
	print("Shape of Outputs: ",rnn_outputs.shape, "shape of states: ", states.shape)
	print("Shape of classifier: ",classifier.shape, "Shape of y: ", y.shape)
	# stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
	# stacked_outputs = fully_connected(stacked_rnn_outputs, n_output, activation_fn=None)
	# outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_output])

	# softmax + cross entropy calculation
	with tf.name_scope('cross_entropy'):
		xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=fc_layer)
		# define loss function & optimize method
		with tf.name_scope('total_loss'):
			loss = tf.reduce_mean(xentropy)
	tf.summary.scalar('xentropy',xentropy)

	with tf.name_scope('training'):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		training_op = optimizer.minimize(loss)

	# measurement
	with tf.name_scope('accuracy'):
		correct = tf.nn.in_top_k(classifier, y, 1)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	tf.summary.scalar('accuracy',accuracy)

	#########################################
	############# Training Phase    #########
	#########################################
	n_epoch = 1#1#5#10

	# split dataset to training set & test set
	n_batch = (len(dataset['close']) + 1 - TIME_SPAN) // BATCH_SIZE
	n_training = int(n_batch * TRAIN_RATIO)
	print("train set: [%d, %d), test set: [%d, %d)" % (0,n_training, n_training, n_batch))

	# Batch Generator. b_index represents batch's index in dataset
	def next_batch(b_index):
		X_batch, y_batch = [],[]
		# retrieve chunks in continuous way. every n_step as a chunk, overlapped
		for i in range(BATCH_SIZE):
		    X_batch.append(dataset['close'].values[b_index+i : b_index+i+TIME_SPAN])
		    y_batch.append(dataset['label'].values[b_index+i+1+TIME_SPAN])   # n time step share 1 label
		return np.array(X_batch).reshape(BATCH_SIZE,TIME_SPAN,n_input), np.array(y_batch)#.reshape(-1)

	train_cnt = 0
	test_cnt = n_training
	def feed_dict(is_training):
		if is_training:
			if train_cnt == n_training:
				train_cnt = 0
			xs, ys = next_batch(train_cnt)
			train_cnt += 1
		else:
			if test_cnt == n_batch:
				test_cnt = n_training
			xs, ys = next_batch(test_cnt)
			test_cnt += 1
		return {X: xs, y: ys}
# yapf: enable
	######### Start Training ########################
	start_time = time.time()
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		# Merge all the summaries and write them out to log dir
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
		test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')    
	
		init.run()
		for j in range(n_epoch):
			best_train_acc, sum_train_acc = 0., 0.
			for i in range(FLAGS.max_steps):
				if i % 5 == 0:
                # Record summaries and test-set accuracy
					summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
					test_writer.add_summary(summary, i)
				  	print('Accuracy at step %s: %s' % (i, acc))
				# Record train set summaries and train
                else:
				  	if i % 10 == 9: # Record execution stats	
						run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
						run_metadata = tf.RunMetadata()
						summary, _ = sess.run([merged, train_step],
						                  feed_dict=feed_dict(True),
						                  options=run_options,
						                  run_metadata=run_metadata)
						train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
						train_writer.add_summary(summary, i)
						print('Adding run metadata for', i)
					# Record a summary
				  	else:
						summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
						train_writer.add_summary(summary, i)
		train_writer.close()
		test_writer.close()

def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	data_prepare()
	train()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--max_steps', type=int, default=100,
		                  help='Number of steps to run trainer.')
	parser.add_argument('--learning_rate', type=float, default=0.01,
		                  help='Initial learning rate')
	parser.add_argument('--dropout', type=float, default=0.9,
		                  help='Keep probability for training dropout.')
	parser.add_argument(
		  '--data_dir',
		  type=str,
		  default='./input_data',
		  help='Directory for storing input data')
	parser.add_argument(
		  '--log_dir',
		  type=str,
		  default='./logs',
		  help='Summaries log directory')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


