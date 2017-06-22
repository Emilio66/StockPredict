from classifier_with_summaries import *
import sys
sys.path.append("..")
from data.dataset.BatchGenerator import *

class param:
	def __init__(self, batch_size=20, data_dir='./input_data', dropout=0.9, input_dim=1, learning_rate=0.001, log_dir='./logs', max_steps=400, n_epoch=4, n_layers=3, n_neurons=150, retrace=0.618, time_steps=10, train_ratio=0.8, use_weight=0):
		self.batch_size=batch_size
		self.data_dir=data_dir
		self.dropout=dropout
		self.input_dim=input_dim
		self.learning_rate=learning_rate
		self.log_dir=log_dir
		self.max_steps=max_steps
		self.n_epoch=n_epoch
		self.n_layers=n_layers
		self.n_neurons=n_neurons
		self.retrace=retrace
		self.time_steps=time_steps
		self.train_ratio=train_ratio
		self.use_weight=use_weight

### run experiments ###
FLAGS = param(batch_size=10, n_epoch=2)
dataset = BatchGenerator('../data/dataset/close_2002-2017.csv',  FLAGS.batch_size, FLAGS.train_ratio,FLAGS.time_steps, FLAGS.input_dim, FLAGS.retrace, fold_i=0, use_weight=FLAGS.use_weight)
acc_train, acc_test = train(dataset, FLAGS)	
print(acc_train, acc_test)
	
