import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import math
def _data_prepare(file_name, retrace = 0.618, no_retrace = False):
    dataset = pd.read_csv(file_name,index_col=0, sep=',', usecols=[0,4], skiprows=1, names=['date','close'],parse_dates=True)
    print(file_name)
    # calculate momentum: Mt = (CLOSE(t) -CLOSE(t-1))/CLOSE(t-1)
    dataset['mmt'] = 0.0
    for i in range(1, len(dataset)):
        dataset['mmt'][i] = (dataset['close'][i] - dataset['close'][i-1]) / dataset['close'][i-1]
    #print("-------------------- Momentum Distribution ------------------")
    #print(dataset['mmt'].value_counts(bins=5).sort_index())
    pd.set_option('mode.chained_assignment',None)
    # classify by counts
    dataset['label'] = 0
    mmt_series = dataset['mmt']
    for i in range(len(dataset)):
        mmt = mmt_series[i]
        if mmt < -0.01: 
            dataset['label'][i] = 0
        elif mmt <= 0.01:
            dataset['label'][i] = 1
        else:
            dataset['label'][i] = 2
    #dataset['norm_close'] = 0.0
    #mean = dataset['close'].mean()
    #for i in range(len(dataset)):
    #	dataset['norm_close'][i] = dataset['close'][i] - mean
    
    dataset['trend']=0
    price = dataset['close']
    size = len(price)
    start = 0
    while price[start] > price[start+1]:
    	start +=1
    print("----- start: ",start)
    #print("-- retrace: ", retrace)
	#find peak, find trough, calculate retracement and label trend accordingly
    i = start
    while i < size - 1:
		cursor = i
		while cursor < size - 1 and price[cursor] <= price[cursor+1]:
			cursor += 1
		peak = cursor
		while cursor < size - 1 and price[cursor] >= price[cursor+1]:
			cursor += 1
		trough = cursor
		retracement = (price[peak] - price[trough]) / (price[peak] - price[i])
		mark = 1 # flat
		if retracement < retrace:
			mark = 2 # UP
		elif retracement > 1 + retrace:
			mark = 0 # DOWN
		for k in range(i, cursor+1):
			dataset['trend'][k] = mark
		i = cursor
    
    #print("---- Label Distribution Check --------")
    #print("Total: ",len(dataset['label']))
    #print(dataset['label'].value_counts().sort_index())
    print("---- Trend Distribution Check --------")
    print(dataset['trend'].value_counts().sort_index())
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , " ------ Complete Data Preparation")
    #dataset['label'].plot(lw=2.0)
    #dataset.plot(lw=2.0)
    #plt.show()
    #print(no_retrace)
    if no_retrace == True:
    	#print('whole data for SVM')
    	return dataset['label']
    else:
    	return dataset['trend'] # only need 1 series

def generateSVMData(file_name, time_step, predict_day=1, no_retrace=False, use_weight=2,  train_ratio=0.8):
	print('params: ',time_step, predict_day, no_retrace, use_weight,train_ratio)
	data = _data_prepare(file_name, no_retrace=no_retrace)
	size = len(data)
	bound = size * train_ratio
	train_x, train_y, test_x, test_y = [],[],[],[]
	#for i in range(0, bound - time_step):
	#	train_x.append(data.values[i : i + time_step])
	#	train_y.append(data.values[i + time_step + predict_day - 1])
	for i in range(0, size - time_step - predict_day):
		test_x.append(weight_assign(data.values[i : i + time_step], use_weight))
		test_y.append(data.values[i + time_step + predict_day - 1])
	return np.array(test_x), np.array(test_y)

# FUNCTION: assign weight by their time point
# NOTICE: use another of original dataset in case of dirty write
def weight_assign(dataseries, use_weight = 0):
	if use_weight > 0:
		if use_weight == 2:
			weight = lambda x: x+1
			
		elif use_weight == 3:
			weight = lambda x: (x+1)*(x+1)
				
		elif use_weight == 4:
			weight = lambda x: (x+1)*(x+1)*(x+1)
		
		elif use_weight == 1:
			weight = lambda x: math.log(x+1)	
			
		elif use_weight == 6:
			weight = lambda x: math.exp(x+1)
			
		elif use_weight == 7: #sigmoid
			weight = lambda x: 1 / (math.exp(-x-1))
			
		elif use_weight == 5: 
			weight = lambda x: (x+1)**4
			
			
		size = len(dataseries)
		copy_dataseries = [0 for i in range(size)]
		for i in range(size):
			copy_dataseries[i] = dataseries[i] * weight(i)
		return copy_dataseries
	return dataseries

###############
# NOTICE:
# We use rolling data here due to the small size of data, label is the data 1 step further into the future
# eg:
# batch0: X=[0,1,2,3] y=[4]
# batch1: X=[1,2,3,4] y=[5]
# 
# Variant K-fold cross-validation specially designed for time-series data are used
# Use 1~K-1 chunk of data as train set, K chunk as test set, size of train set keep growing
# eg:
# train1: X=[0,1,2,3] test1=[4]
# train2: X=[0,1,2,3,4] test2=[5]
# train3: X=[0,1,2,3,4,5] test3=[6]
class BatchGenerator(object):
	def __init__(self, file_name, batch_size, train_ratio, time_steps, input_size, retrace = 0.618, fold_i=0, use_weight=False, no_retrace=False):
		self.train_ratio = train_ratio
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.input_size = input_size
		self.no_retrace = no_retrace
		self.dataset = _data_prepare(file_name, retrace, no_retrace=no_retrace)
		#self.segment_num = (len(self.dataset)) // batch_size // time_steps
		self.segment_num = (len(self.dataset) - time_steps) // batch_size # rollingly use data
		
		# for K-fold cross-validation, split train and test data evenly by parameter train_ratio
		# fold_i indicate the index of data chunk (default 10 fold for cross-validation)
		self.train_size = int(self.segment_num * train_ratio)
		remainder = self.segment_num - self.train_size
		if remainder >= 2:
			self.test_size = remainder / 2
			self.buffer_size = remainder - self.test_size
		else:
			self.test_size = remainder 
			self.buffer_size = 0
		fold_i = fold_i % (self.buffer_size + 1)
		self.train_dataset = self.dataset.values[0 : (self.train_size + fold_i) * self.batch_size + self.time_steps + 1]
		self.test_dataset = self.dataset.values[(self.train_size + fold_i) * self.batch_size : (self.train_size + self.test_size + fold_i) * self.batch_size + self.time_steps + 1]
		self.fold_i = fold_i
		self.train_cursor = 0
		self.test_cursor = 0
		self.use_weight = use_weight

		print("Training set size: ", len(self.train_dataset)," Test set size: ", len(self.test_dataset))
		#print("Batch Size: ", batch_size, "Train Batch Num: ", self.train_size,"Test Batch Num: ", self.test_size," Buffer Batch Num: ",self.buffer_size)
		#print("Train interval: ",0, (self.train_size + fold_i) * self.batch_size + self.time_steps + 1)
		#print("Test interval: ",(self.train_size + fold_i) * self.batch_size, (self.train_size + self.test_size + fold_i) * self.batch_size + self.time_steps + 1)
	
	# generate train&test dataset for cross-validation
	def resetFold(self, fold_i):
		fold_i = fold_i % (self.buffer_size + 1)
		self.train_dataset = self.dataset.values[0 : (self.train_size + fold_i) * self.batch_size + self.time_steps + 1]
		self.test_dataset = self.dataset.values[(self.train_size + fold_i) * self.batch_size : (self.train_size + self.test_size + fold_i) * self.batch_size + self.time_steps + 1]
		self.fold_i = fold_i
		self.train_cursor = 0
		self.test_cursor = 0
	
	@property
	def max_fold(self):
		return self.buffer_size
	
	def _next_batch(self, b_index, isTraining = True):
		X_batch, y_batch = [],[]
		# retrieve chunks in continuous way. every instance is independent
		if isTraining == True:
			datasource = self.train_dataset
		else:
			datasource = self.test_dataset
		for i in range(self.batch_size ):
			#X_batch.append(self.dataset['norm_close'].values[b_index + i * self.time_steps : b_index + (i + 1) * self.time_steps])
			#y_batch.append(self.dataset['label'].values[b_index + (i + 1) * self.time_steps])
			data_series_x = datasource[b_index + i : b_index + i + self.time_steps]
			X_batch.append(weight_assign(data_series_x, self.use_weight))#time-weighted processing
			y_batch.append(datasource[b_index + i + self.time_steps])
		return np.array(X_batch).reshape(self.batch_size, self.time_steps, self.input_size), np.array(y_batch)
		
	# Batch Generator. b_index represents batch's index in dataset
	#def _next_batch(self, b_index):
	#	X_batch, y_batch = [],[]
		# retrieve chunks in continuous way. every n_step as a chunk, overlapped
	#	for i in range(self.batch_size ):
	#		X_instance, y_instance=[],[]
	#		for j in range(self.time_steps):
	#			X_instance.append(self.dataset['close'].values[b_index + i * self.time_steps + j : b_index + i * self.time_steps + j + self.input_size])
	#			y_instance.append(self.dataset['label'].values[b_index + i * self.time_steps + j + self.input_size]) # every time step got a label
	#		X_batch.append(X_instance)
	#		y_batch.append(y_instance)   
	#	return np.array(X_batch), np.array(y_batch)
	
	def next_batch(self, is_training=True):
		if is_training:
			#xs, ys = self.next_batch(self.train_cursor * self.batch_size * self.time_steps)
			xs, ys = self._next_batch(self.train_cursor * self.batch_size, is_training) #rolling use
			self.train_cursor = (self.train_cursor + 1) % (self.train_size + self.fold_i)
		else:
			#if self.test_cursor >= self.segment_num:
			#	self.test_cursor =  self.train_size
			#print("BOUND: ", self.test_cursor * self.batch_size * self.time_steps)
			xs, ys = self._next_batch(self.test_cursor * self.batch_size, is_training)
			self.test_cursor = (self.test_cursor + 1) % self.test_size
		dic = {}
		dic['X'] = xs
		dic['y'] = ys
		return dic

#### Unit Test ######
if __name__ == '__main__':
	#generator = BatchGenerator('close_2012-2017.csv', batch_size=10, train_ratio=0.9, time_steps=4, input_size=1)
	#BatchGenerator('../data/dataset/close_2002-2017.csv',  FLAGS.batch_size, FLAGS.train_ratio,FLAGS.time_steps, FLAGS.input_dim, FLAGS.retrace, fold_i=0, use_weight=FLAGS.use_weight)
	generator = BatchGenerator('close_weekly-2007-2017.csv', 10, 0.9, 16, 1)
	#print(generator.next_batch())
	#print("TRAIN CURSOR: ", generator.train_cursor)
	#print(generator.next_batch())
	#print("TRAIN CURSOR: ", generator.train_cursor)
	#print(generator.next_batch(is_training=False))
	#print("TEST CURSOR: ", generator._test_cursor)
	for i in range(0,10):
		#print(i)
		generator.next_batch(is_training=False)
	#print(generator.next_batch(is_training=False))
	#print("TEST CURSOR: ", generator.test_cursor)
	#generator.next_batch(is_training=False)
	#generator.next_batch()
	#generator.next_batch()
	
	


