from classifier_with_GRU import *
import sys
sys.path.append("..")
from data.dataset.BatchGenerator import *
import time

class param:
	def __init__(self, batch_size=10, data_dir='./input_data', dropout=0.9, input_dim=1, learning_rate=0.001, log_dir='./logs', max_steps=400, n_epoch=4, n_layers=3, n_neurons=150, retrace=0.618, time_steps=4, train_ratio=0.8, use_weight=0, no_retrace=False, fine_grained = False):
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
		self.no_retrace = no_retrace
		self.fine_grained = fine_grained

	def __str__(self):
		return "batch_size "+str(self.batch_size) +" retrace: "+ str(self.retrace)+ " epoch: "+str(self.n_epoch)+" weight: "+	str(self.use_weight) + " n_layers: "+str(self.n_layers)+" n_neurons "+ str(self.n_neurons)+" time_steps: "+str(self.time_steps)+" max_steps "+str(self.max_steps) + " learning rate: "+str(self.learning_rate)
### run experiments ###
#FLAGS = param(batch_size=10, n_epoch=2)
#dataset = BatchGenerator('../data/dataset/close_weekly-2007-2017.csv',  FLAGS.batch_size, FLAGS.train_ratio,FLAGS.time_steps, FLAGS.input_dim, FLAGS.retrace, fold_i=0, use_weight=FLAGS.use_weight)

#dataset = BatchGenerator('../data/dataset/close_2002-2017.csv',  FLAGS.batch_size, FLAGS.train_ratio,FLAGS.time_steps, FLAGS.input_dim, FLAGS.retrace, fold_i=0, use_weight=FLAGS.use_weight)

#print(acc_train, acc_test)
####### time steps experiment #####
'''
size = 20
avg = 2
filename = './exp/weekly_time_step1-'+str(size)+'.txt'

lines = []
for i in range(1, size+1):
	FLAGS.time_steps = i
	dataset.time_steps = i
		# on average
	sum_train, sum_test = 0., 0.
	for j in range(0, avg):
		acc_train, acc_test = train(dataset, FLAGS)
		sum_train += acc_train
		sum_test += acc_test
	line  = str(i) + '\t' +str(sum_train/avg) +'\t' +str(sum_test/avg) +'\r\n'
	lines.append(line)
with open(filename, 'w') as f:
	f.writelines(lines)
print("Wring file", filename, "complete!")

############ retracement test ###
maxR = 0.69
minR = 0.33
delta = 0.05
filename = './exp/weekly_retracement-5'+str(minR)+'-'+str(maxR)+'.txt'
retrace = minR
FLAGS = param(batch_size=10, n_epoch=3)
dataset = BatchGenerator('../data/dataset/close_weekly-2007-2017.csv',  FLAGS.batch_size, FLAGS.train_ratio,FLAGS.time_steps, FLAGS.input_dim, retrace, fold_i=0, use_weight=FLAGS.use_weight)
lines = []
while retrace <= maxR:
	FLAGS.retrace = retrace
	dataset.retrace = retrace
	acc_train, acc_test = train(dataset, FLAGS)
	line  = str(retrace) +'\t' +str(acc_test) + '\t' +str(acc_train) +'\n'
	lines.append(line)
	retrace += delta
with open(filename, 'w') as f:
	f.writelines(lines)
print("Wring file", filename, "complete!")
'''
############ weight function test ###
print (sys.argv)
if sys.argv[1] == '0':
	num = 7
	avg = 3

	FLAGS = param(time_steps=4, n_neurons=320, learning_rate=0.0024, no_retrace=False)
	dataset = BatchGenerator('../data/dataset/close_2007-2017.csv',  FLAGS.batch_size, FLAGS.train_ratio,FLAGS.time_steps, FLAGS.input_dim, FLAGS.retrace, fold_i=0, use_weight=FLAGS.use_weight)
	lines = []
	filename = './exp/weighted_function_GRU4'+str(num)+'.txt'

	for i in range(num+1):
		FLAGS.use_weight = i
		dataset.use_weight = i
		sum_train, sum_test = 0., 0.
		#for j in range(0, avg):
		acc_train, acc_test = train(dataset, FLAGS)
			#sum_train += acc_train
			#sum_test += acc_test
		#line  = str(i) + '\t' +str(sum_train/avg) +'\t' +str(sum_test/avg) +'\n'
		line  = str(i) +'\t' +str(acc_test) + '\t' +str(acc_train) +'\n'
		lines.append(line)
	with open(filename, 'w') as f:
		f.writelines(lines)
	print("Writing file", filename, "complete!")

################# n_neurons test ####
elif sys.argv[1] == '1':
	maxS = 340#500
	minS = 280#50
	step = 10 #50

	FLAGS = param(batch_size=20, n_epoch=3, )
	dataset = BatchGenerator('../data/dataset/close_weekly-2007-2017_2.csv',  FLAGS.batch_size, FLAGS.train_ratio,FLAGS.time_steps, FLAGS.input_dim, FLAGS.retrace, fold_i=0, use_weight=FLAGS.use_weight, no_retrace=True)
	lines = []
	filename = './exp/weekly_n_neurons_'+str(minS)+'-'+str(maxS)+'.txt'
	for i in range(minS, maxS, step):
		FLAGS.n_neurons = i
		dataset.n_neurons = i
		acc_train, acc_test = train(dataset, FLAGS)
		line  = str(i) +'\t' +str(acc_test) + '\t' +str(acc_train) +'\n'
		lines.append(line)
	with open(filename, 'w') as f:
		f.writelines(lines)
	print("Writing file", filename, "complete!")

################# n_layers test ####
elif sys.argv[1] == '2':
	maxS = 20
	minS = 1
	step = 1

	FLAGS = param(batch_size=20, n_epoch=3)
	dataset = BatchGenerator('../data/dataset/close_weekly-2007-2017.csv',  FLAGS.batch_size, FLAGS.train_ratio,FLAGS.time_steps, FLAGS.input_dim, FLAGS.retrace, fold_i=0, use_weight=FLAGS.use_weight)
	lines = []
	filename = './exp/weekly_layers_n_'+str(minS)+'-'+str(maxS)+'.txt'
	for i in range(minS, maxS, step):
		FLAGS.n_layers = i
		dataset.n_layers = i
		acc_train, acc_test = train(dataset, FLAGS)
		line  = str(i) +'\t' +str(acc_test) + '\t' +str(acc_train) +'\n'
		lines.append(line)
	with open(filename, 'w') as f:
		f.writelines(lines)
	print("Writing file", filename, "complete!")

#### Learning rate test ############
elif sys.argv[1] == '3':
	maxS = 0.0030#0.1
	minS = 0.0008#0.0001
	step = 0.0002

	FLAGS = param(batch_size=20, n_epoch=3)
	dataset = BatchGenerator('../data/dataset/close_weekly-2007-2017.csv',  FLAGS.batch_size, FLAGS.train_ratio,FLAGS.time_steps, FLAGS.input_dim, FLAGS.retrace, fold_i=0, use_weight=FLAGS.use_weight)
	lines = []
	filename = './exp/weekly_learning_rate_'+str(minS)+'-'+str(maxS)+'.txt'
	i = minS
	while i < maxS :
		FLAGS.learning_rate = i
		dataset.learning_rate = i
		acc_train, acc_test = train(dataset, FLAGS)
		line  = str(i) +'\t' +str(acc_test) + '\t' +str(acc_train) +'\n'
		i = i+step
		lines.append(line)
	with open(filename, 'w') as f:
		f.writelines(lines)
	print("Writing file", filename, "complete!")

#### max training test ############
elif sys.argv[1] == '4':
	maxS = 5500#2500#10000
	minS = 2500#0.0001
	step = 500

	FLAGS = param(batch_size=20, n_epoch=4, n_neurons=320, learning_rate=0.0024)
	dataset = BatchGenerator('../data/dataset/close_weekly-2007-2017.csv',  FLAGS.batch_size, FLAGS.train_ratio,FLAGS.time_steps, FLAGS.input_dim, FLAGS.retrace, fold_i=0, use_weight=FLAGS.use_weight)
	lines = []
	filename = './exp/weekly_max_steps_'+str(minS)+'-'+str(maxS)+'.txt'
	i = minS
	while i <= maxS :
		FLAGS.max_steps = i
		dataset.max_steps = i
		acc_train, acc_test = train(dataset, FLAGS)
		line  = str(i) +'\t' +str(acc_test) + '\t' +str(acc_train) +'\n'
		i = i+step
		lines.append(line)
	with open(filename, 'w') as f:
		f.writelines(lines)
	print("Writing file", filename, "complete!")

####### Fine-grained weight-function exploration
elif sys.argv[1] == '5':
	maxP = 3.0
	minP = 0.9
	step = 0.1

	FLAGS = param(time_steps=8, n_neurons=320, n_epoch=2, learning_rate=0.0024, no_retrace=False, fine_grained= True)
	dataset = BatchGenerator('../data/dataset/close_2007-2017.csv',  FLAGS.batch_size, FLAGS.train_ratio,FLAGS.time_steps, FLAGS.input_dim, FLAGS.retrace, fold_i=0, use_weight=FLAGS.use_weight, fine_grained = True)
	lines = []
	filename = './exp/fine_grained_weighted_function-step-'+str(step)+'.txt'
	i = minP
	while i <= maxP:
		FLAGS.use_weight = i
		dataset.use_weight = i
		sum_train, sum_test = 0., 0.
		#for j in range(0, avg):
		acc_train, acc_test = train(dataset, FLAGS)
			#sum_train += acc_train
			#sum_test += acc_test
		#line  = str(i) + '\t' +str(sum_train/avg) +'\t' +str(sum_test/avg) +'\n'
		line  = str(i) +'\t' +str(acc_test) + '\t' +str(acc_train) +'\n'
		lines.append(line)
		i += step
	with open(filename, 'w') as f:
		f.writelines(lines)
	print("Writing file", filename, "complete!")

elif sys.argv[1] == '6':
	cols = [1,2,4,5,6]
	weight = 3	#nlogn
	lines = []
	filename = './exp/final_weighted_GRU_datasets.txt'
	for i in cols:
		FLAGS = param(time_steps=4, n_neurons=320, learning_rate=0.0024, no_retrace=False)
		dataset = BatchGenerator('../data/dataset/close_2007-2017.csv', FLAGS.batch_size, FLAGS.train_ratio,FLAGS.time_steps, FLAGS.input_dim,column = i, retrace=FLAGS.retrace, fold_i=0, use_weight=weight)
		acc_train, acc_test = train(dataset, FLAGS)
		line  = str(i) +'\t' +str(acc_test) + '\t' +str(acc_train) +'\n'
		lines.append(line)
	with open(filename, 'w') as f:
		f.writelines(lines)
	print("Writing file", filename, "complete!")
