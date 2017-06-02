''' Unit Test Whether RNN Works As Expected '''

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os

sys.path.append("..")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from data.dataset.BatchGenerator import *

def rnn(dataset, n_steps, n_input, n_neurons, n_layers, n_output, learning_rate, n_iteration):
	###### contruction phase ###
	with tf.name_scope('input'):
		X = tf.placeholder(tf.float32, [None, n_steps, n_input], name='X-input')
		y = tf.placeholder(tf.int32, [None, n_steps], name='y-input') # every time step correspond to a label

	
	with tf.variable_scope("lstm_layers", initializer=tf.contrib.layers.variance_scaling_initializer()):
		lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons, use_peepholes=False)
		cells = tf.contrib.rnn.MultiRNNCell([lstm_cell]*n_layers)
		rnn_outputs, states = tf.nn.dynamic_rnn(cells, X, dtype=tf.float32)
	
	fc_layer = fully_connected(rnn_outputs, n_output, activation_fn=None)
	
	with tf.name_scope('cross_entropy'):
		xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=fc_layer)
		print("Shape Cross-entropy", xentropy.shape)
		with tf.name_scope('total_loss'):
			loss = tf.reduce_mean(xentropy)
			print("Shape LOSS:", loss.shape)

	with tf.name_scope('training'):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		training_op = optimizer.minimize(loss)

	print("Shape of Outputs: ",rnn_outputs.shape)
	print("Shape of states: ", tf.shape(states))
	print("Shape of fc_layer: ",fc_layer.shape)
	print("Shape of y: ", y.shape)
	
	############# Training Phase    #########
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		init.run()
		for i in range(n_iteration):
			data = dataset.next_batch()
			_, out, st, ls = sess.run([training_op, rnn_outputs, states,loss],feed_dict={X: data['X'], y: data['y']})
			#print('DATA: ')
			#print(data)
			print('RNN_OUTPUT: ')
			print(out)
			print('RNN_STATES: ')
			print(st)
			print("loss: ")
			print(ls)
	
if __name__ == '__main__':
	n_batch = 3
	n_steps = 4
	n_input = 1
	n_neurons = 6
	n_layers = 3
	learning_rate = 0.001
	n_output = 3
	n_iteration = 10
	
	dataset = BatchGenerator('../data/dataset/close_2012-2017.csv', n_batch, 0.9, n_steps, n_input)
	rnn(dataset,n_steps, n_input, n_neurons, n_layers, n_output, learning_rate, n_iteration)
	
