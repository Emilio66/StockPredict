# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

data_file = "..\\data\\dataset\\close_2016-2017.csv"
dataset = pd.read_csv(data_file,index_col=0, sep=',', usecols=[0,1], skiprows=1, names=['date','close'],parse_dates=True)
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , " ------ Complete Data Reading")
# how long will a span cover, e.g. 20 days
TIME_SPAN = 20
TRAIN_RATIO = 0.9
BATCH_SIZE = 3

# y batch is just 1 step further into the future than X batch
def next_batch(n_batch, n_step, index=0):
    X_batch, y_batch = [],[]
    for i in range(index, n_batch+index):
        X_batch.append(dataset['close'].values[i : i+n_step])
        y_batch.append(dataset['close'].values[i+1 : i+1+n_step])
    return X_batch, y_batch

for i in range(1):
    X_batch, y_batch = next_batch(BATCH_SIZE, TIME_SPAN, i*TIME_SPAN*BATCH_SIZE)

print(type(X_batch)) #  <class 'list'>
print("-------- \n ",type(X_batch[0]), len(X_batch[0])," \n--------\n", X_batch[0]) # <class 'numpy.ndarray'>

#########################
### contruction phase ###
#########################
tf.reset_default_graph()
n_neurons = 20
n_steps = 20
n_input = 1
n_output = 1
n_layers = 3
learning_rate = 0.001#0.0005 # # 0.02 # 0.005

X = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_steps, n_output])

# define network, 1 layer now
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.relu)
# 3 layers' lstm
#multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*3)
#rnn_outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
# Add Dropout
is_training = False 
keep_prob = 0.75
if is_training:
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob)
rnn_outputs, states = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = fully_connected(stacked_rnn_outputs, n_output, activation_fn=None)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_output])
# define loss optimizer
loss = tf.reduce_sum(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

##############################
##### Training Phase #########
##############################
n_epoch = 10
data_size = len(dataset['close']) // BATCH_SIZE
n_training = int(data_size * TRAIN_RATIO)

# add a test case
index = n_training*TIME_SPAN*BATCH_SIZE
X_test, y_test = next_batch(BATCH_SIZE,TIME_SPAN, index)
date_axis = dataset['date'].values[index : index + TIME_SPAN + 1]

start_time = time.time()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(n_training):
        X_batch, y_batch = next_batch(BATCH_SIZE, TIME_SPAN, i*TIME_SPAN*BATCH_SIZE)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if i % 10 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(i,elapsed(time.time()-start_time), "MSE: ", mse)
    # evaluate
    # X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_input)))
    y_pred = sess.run(outputs, feed_dict={X: X_test})
    #print(y_pred)
    #y_target = time_series(np.array(t_instance[1:].reshape(-1, n_steps, n_input)))
    #print(y_target)

plt.title("Testing the model", fontsize=14)
plt.plot(date_axis[:-1], X_test, "bo", markersize=10, label="instance")
plt.plot(date_axis[1:], y_test, "y*", markersize=10, label="target")
plt.plot(date_axis[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.show()