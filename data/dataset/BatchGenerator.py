import numpy as np
import pandas as pd
import time

def _data_prepare(file_name):
    dataset = pd.read_csv(file_name,index_col=0, sep=',', usecols=[0,1], skiprows=1, names=['date','close'],parse_dates=True)

    # calculate momentum: Mt = (CLOSE(t) -CLOSE(t-1))/CLOSE(t-1)
    dataset['mmt'] = 0.0
    for i in range(1, len(dataset)):
        dataset['mmt'][i] = (dataset['close'][i] - dataset['close'][i-1]) / dataset['close'][i-1]
    print("-------------------- Momentum Distribution ------------------")
    print(dataset['mmt'].value_counts(bins=5).sort_index())
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
    print("---- Label Distribution Check --------")
    print("Total: ",len(dataset['label']))
    print(dataset['label'].value_counts().sort_index())
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , " ------ Complete Data Preparation")
    return dataset

class BatchGenerator(object):
	def __init__(self, file_name, batch_size, train_ratio, time_steps, input_size):
		self._train_ratio = train_ratio
		self._batch_size = batch_size
		self._time_steps = time_steps
		self._input_size = input_size
		self._dataset = _data_prepare(file_name)
		self._segment_num = (len(self._dataset) - input_size) // batch_size // time_steps
		self._train_bound = int(self._segment_num * train_ratio)
		self._train_cursor = 0
		self._test_cursor = self._train_bound
		print("DATASET FOR TRAINING: [0", self._train_bound, "), TEST SET: [", self._train_bound, self._segment_num," )")

	# Batch Generator. b_index represents batch's index in dataset
	def _next_batch(self, b_index):
		X_batch, y_batch = [],[]
		# retrieve chunks in continuous way. every n_step as a chunk, overlapped
		for i in range(self._batch_size ):
			X_instance, y_instance=[],[]
			for j in range(self._time_steps):
				X_instance.append(self._dataset['close'].values[b_index + i * self._time_steps + j : b_index + i * self._time_steps + j + self._input_size])
				y_instance.append(self._dataset['label'].values[b_index + i * self._time_steps + j + self._input_size]) # every time step got a label
			X_batch.append(X_instance)
			y_batch.append(y_instance)   
		return np.array(X_batch), np.array(y_batch)
	
	def next_batch(self, is_training=True):
		if is_training:
			xs, ys = self._next_batch(self._train_cursor * self._batch_size * self._time_steps)
			self._train_cursor = (self._train_cursor + 1) % self._train_bound
		else:
			if self._test_cursor >= self._segment_num:
				self._test_cursor =  self._train_bound
			xs, ys = self._next_batch(self._test_cursor * self._batch_size * self._time_steps)
			self._test_cursor = self._test_cursor + 1
		dic = {}
		dic['X'] = xs
		dic['y'] = ys
		return dic

#### Unit Test ######
if __name__ == '__main__':
	generator = BatchGenerator('close_2012-2017.csv', batch_size=10, train_ratio=0.9, time_steps=4, input_size=3)
	generator.next_batch()
	print("TRAIN CURSOR: ", generator._train_cursor)
	print(generator.next_batch())
	print("TRAIN CURSOR: ", generator._train_cursor)
	print(generator.next_batch(is_training=False))
	print("TEST CURSOR: ", generator._test_cursor)
	print(generator.next_batch(is_training=False))
	print("TEST CURSOR: ", generator._test_cursor)
	
	


