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
import sys
sys.path.append("..")
from data.dataset.BatchGenerator import *

# yapf: disable
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
FLAGS = None

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
def train(dataset, FLAGS):
	tf.reset_default_graph()

	# how long will a span cover, e.g. 20 days (4 tradable weeks)
	TIME_SPAN = FLAGS.time_steps # 4weeks, 1month
	TRAIN_RATIO = 0.9#0.8
	BATCH_SIZE = FLAGS.batch_size
	n_neurons = FLAGS.n_neurons #250
	n_steps = TIME_SPAN
	n_input = FLAGS.input_dim #5, 1 week's data as feature
	n_output = 3#5 #5 class
	n_layers = FLAGS.n_layers #5#10 #5 #3
	learning_rate = FLAGS.learning_rate #0.0005 # # 0.02 # 0.005
	print(FLAGS)
	
	with tf.name_scope('input'):
		X = tf.placeholder(tf.float32, [None, n_steps, n_input], name='X-input')
		#y = tf.placeholder(tf.int32, [None, n_steps], name='y-input') # every time step correspond to a label
		y = tf.placeholder(tf.int32, [None], name='y-input') # every instance relates to 1 label

	
	with tf.variable_scope("lstm_layers", initializer=tf.contrib.layers.variance_scaling_initializer()):
		# define network, 3 LSTM layer now, use tf.tanh as activation function, use peephole
		#lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons, use_peepholes=False)
		#lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons, activation=tf.nn.relu, use_peepholes=False)
		stacked_cells = []
		for i in range(n_layers):
			stacked_cells.append(tf.contrib.rnn.LSTMCell(num_units=n_neurons))
		cells = tf.contrib.rnn.MultiRNNCell(stacked_cells)
		rnn_outputs, states = tf.nn.dynamic_rnn(cells, X, dtype=tf.float32)

	# Add Dropout
	#is_training = False 
	#keep_prob = 0.75
	#if is_training:
	#    lstm_cell = tf.contrib.rnn.DropoutWrapper(cells, input_keep_prob=keep_prob)

	#print('cells.output_size: ', cells.output_size)
	#print('cells.state_size: ',cells.state_size)
	#print("STATES: ", states)

	tf.summary.histogram('lstm_outputs',rnn_outputs) # new_h, every output
	#tf.summary.histogram('lstm_states', states)  # state is just the tuple(new_c, new_h)

	states = states[-1][1] #retrieve the last layer's state tuple and only need last output/hypothesis states, omit memory cell
	#print("Final STATES: ", states)
	tf.summary.histogram('lstm_cell_states', states)
	#states = tf.concat(axis=1, values=states) #sum up all neuron's result at final step
	#tf.summary.histogram('lstm_cell_states_plus_bias', states)

	#with tf.name_scope('fully_connected_layer'):
	fc_layer = fully_connected(states, n_output, activation_fn=None)
	#fc_layer = fully_connected(rnn_outputs, n_output, activation_fn=None)

	#print("RNN Outputs: ",rnn_outputs, " RNN states: ", states) 	# Output [batch_size, n_steps, n_neurons], States[batch_size, n_neurons]
	#print("Shape of Outputs: ",rnn_outputs.shape)
	#print(" states: ", states)
	#print("Shape of fc_layer: ",fc_layer.shape)
	#print("Shape of y: ", y.shape) # FC_Layer [batch_size, n_steps, n_outputs], Label [batch_size, n_steps]
	#print("FC Layer last output:",fc_layer[:,:,-1])
	# stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
	# stacked_outputs = fully_connected(stacked_rnn_outputs, n_output, activation_fn=None)
	# outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_output])

	# softmax + cross entropy calculation
	with tf.name_scope('cross_entropy'):
		xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=fc_layer)
		#print("Cross-entropy", xentropy)	#[batch_size, n_steps], float32
		# define loss function & optimize method
		with tf.name_scope('total_loss'):
			loss = tf.reduce_mean(xentropy)
			#print("LOSS:", loss)
	tf.summary.scalar('xentropy_mean', loss)

	with tf.name_scope('training'):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		training_op = optimizer.minimize(loss)

	# measurement
	with tf.name_scope('accuracy'):
		####correct = tf.nn.in_top_k(fc_layer[:,:,-1], y[:,-1], 1) 	# only compare the final state's output class with label
		#final_predict = tf.slice(fc_layer,[0, n_steps - 1, 0],[-1, 1, -1]) 	# [batch_size, n_outputs]
		#final_label = tf.slice(y,[0, n_steps - 1], [-1, 1])					# [batch_size] (1 label)
		#print("sliced predict: ", final_predict, "sliced label:", final_label)
		#final_predict = tf.reshape(final_predict,[-1, n_output])
		#final_label = tf.reshape(final_label,[-1])
		#print("FINAL predict: ", final_predict, "FINAL label:", final_label)
		#correct = tf.nn.in_top_k(final_predict, final_label, 1) # accuracy in top k
		correct = tf.nn.in_top_k(fc_layer, y, 1)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	tf.summary.scalar('accuracy_mean',accuracy)

	#########################################
	############# Training Phase    #########
	#########################################
	n_epoch = FLAGS.n_epoch
	if n_epoch > dataset.max_fold:
		n_epoch = dataset.max_fold	# valid K-fold CV
	
	# yapf: enable
	######### Start Training ########################
	start_time = time.time()
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		# Merge all the summaries and write them out to log dir
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
		test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')    
	
		avg_train, avg_test = [0], [0]
		sum_train, sum_test =0., 0. #for all iterations
		train_res, test_res = [], []
		
		init.run()
		for j in range(n_epoch):
			dataset.resetFold(j)
			best_train_acc, sum_train_acc = 0., 0.
			best_test_acc, sum_test_acc = 0., 0.
			test_freq = 10
			
			for i in range(FLAGS.max_steps):	
				if i % test_freq == 0:
					# Record summaries and test-set accuracy
					data = dataset.next_batch(is_training=False)
					summary, acc_test = sess.run([merged, accuracy], feed_dict={X: data['X'], y: data['y']})
					test_writer.add_summary(summary, i+j*FLAGS.max_steps)
					#print('Test Accuracy %s: %s' % (i, acc_test))
					if acc_test > best_test_acc:
						best_test_acc = acc_test
					sum_test_acc += acc_test
					sum_test += acc_test
					avg_test.append(sum_test/(len(avg_test)+1))
					#test_res.append(acc_test)
				# Record train set summaries and train
				elif i % 100 == 99: # Record execution stats	
					data = dataset.next_batch()
					run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
					run_metadata = tf.RunMetadata()
					summary, acc_train, _ = sess.run([merged, accuracy, training_op],
					#summary, _, acc, fp, fl, cr, out, st = sess.run([merged, training_op, accuracy,final_predict,final_label,correct,rnn_outputs, states],
						                  feed_dict={X: data['X'], y: data['y']},
						                  options=run_options,
						                  run_metadata=run_metadata)
					train_writer.add_run_metadata(run_metadata, 'step%03d' % (i+j*FLAGS.max_steps))
					train_writer.add_summary(summary, i+j*FLAGS.max_steps)
					#print('Adding run metadata for %d, Acc: %s' % (i, acc_train))
					sum_train_acc += acc_train
					# add average training accuracy
					sum_train += acc_train
					avg_train.append(sum_train/(len(avg_train)+1))
					#train_res.append(acc_train)
					#print('FC Layer value: ', fc)
					#print('Final Predict value: ', fp)
					#print('Label value: ', lb)
					#print('Final Label value: ', fl)
					#print('CORRECT: ', cr)
					#print('RNN_OUTPUT: ', out)
					#print('RNN_STATES: ', st)
					# Record a summary
				else:
					data = dataset.next_batch()
					summary, _, acc_train = sess.run([merged, training_op, accuracy], feed_dict={X: data['X'], y: data['y']})
					train_writer.add_summary(summary, i+j*FLAGS.max_steps)
					if acc_train > best_train_acc:
						best_train_acc = acc_train
					sum_train_acc += acc_train
					# add average training accuracy
					sum_train += acc_train
					avg_train.append(sum_train/(len(avg_train)+1))
					#train_res.append(acc_train)
				
			print(" ========= EPOCH %d STATISTICS============" % (j+1))
			print("   BEST Train Accuracy:", best_train_acc, " AVERAGE Train Accuracy:", sum_train_acc/FLAGS.max_steps)
			print("   BEST Test Accuracy:", best_test_acc, " AVERAGE Test Accuracy:", sum_test_acc/FLAGS.max_steps*test_freq)
			print(" ========= ========= ========= =========")
		train_writer.close()
		test_writer.close()
		x_test = [i for i in range(0, FLAGS.max_steps*n_epoch, test_freq-1)]
		size = len(avg_train)
		
		print("==== FINAL TRAINING ACC: ", avg_train[-1])
		print("==== FINAL TESTING ACC: ", avg_test[-1])
		print(" ========= ========= ========= =========")
		#plt.title("Average Accuracy of the Model", fontsize=34)
		plt.plot(avg_train, markersize=12, linewidth=3, label="Train")
		plt.plot(x_test[: len(avg_test)], avg_test, markersize=12, linewidth=3, label="Test")
		plt.legend(loc="upper left")
		plt.xlabel("Steps", fontsize=16)
		plt.ylabel("Average Accuracy", fontsize=16)
		#plt.show()
		return avg_train[-1], avg_test[-1]
		
def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	filename = '../data/dataset/close_weekly-2007-2017.csv'
	#filename = '../data/dataset/close_2002-2017.csv'
	#filename = '../data/dataset/Dow Jones Industrial Average.csv'
	#filename = '../data/dataset/S&P 500.csv'
	dataset = BatchGenerator(filename,  FLAGS.batch_size, FLAGS.train_ratio,FLAGS.time_steps, FLAGS.input_dim, FLAGS.retrace, fold_i=0, use_weight=FLAGS.use_weight)
	train(dataset, FLAGS)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--max_steps', type=int, default=400,
		                  help='Number of steps to run trainer.')
	parser.add_argument('--learning_rate', type=float, default=0.001,
		                  help='Initial learning rate')
	parser.add_argument('--batch_size', type=int, default=20,
		                  help='number of instances in a batch')
	parser.add_argument('--time_steps', type=int, default=4,
		                  help='Number of time steps.')
	parser.add_argument('--input_dim', type=int, default=1,
		                  help='Dimension of inputs.')
	parser.add_argument('--n_neurons', type=int, default=150,
		                  help='Number of neurons.')
	parser.add_argument('--n_layers', type=int, default=3,
		                  help='Number of lstm layers.')
	parser.add_argument('--train_ratio', type=float, default=0.8,
		                  help='Keep probability for training dropout.')	                  
	parser.add_argument('--dropout', type=float, default=0.9,
		                  help='Keep probability for training dropout.')
	parser.add_argument('--retrace', type=float, default=0.618,
		                  help='Retracement of wave.')
	parser.add_argument('--use_weight', type=float, default=0.0,
		                  help='Whether use time-weighted function.')               
	parser.add_argument('--n_epoch', type=int, default=4,
		                  help='Epoch Number.')
	parser.add_argument('--no_retrace', type=bool, default=False,
		                  help='Whether use retracement.')
	parser.add_argument('--fine_grained', type=bool, default=False,
		                  help='Use fine-grained weight function.')
		                  
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


