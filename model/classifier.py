# -*- coding: UTF-8 -*-
# import tensorflow as tf
# from tensorflow.contrib.layers import fully_connected
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import time

data_file = "..\\data\\dataset\\close_2016-2017.csv"
dataset = pd.read_csv(data_file,index_col=0, sep=',', usecols=[0,1], skiprows=1, names=['date','close'],parse_dates=True)
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , " ------ Complete Data Reading")
# how long will a span cover, e.g. 20 days
TIME_SPAN = 20
TRAIN_RATIO = 0.85
BATCH_SIZE = 3
# 用HISTORY_DAYS天数据来预测未来PREDICT_DAYS天之后为True，跌为False
# labels = dataset['close'].values[PREDICT_DAYS:] > dataset['close'].values[:-PREDICT_DAYS]
# labels = labels[HISTORY_DAYS:]

# y batch is just 1 step further into the future than X batch
def next_batch(n_batch, n_step, index=0):
    X_batch, y_batch = [],[]
    for i in range(index, n_batch+index):
        X_batch.append(dataset['close'].values[i : i+n_step])
        y_batch.append(dataset['close'].values[i+1 : i+1+n_step])
    return X_batch, y_batch

n_iterations = len(dataset['close']) // BATCH_SIZE
for i in range(1):
    X_batch, y_batch = next_batch(BATCH_SIZE, TIME_SPAN, i*TIME_SPAN*BATCH_SIZE)

print(type(X_batch)) #  <class 'list'>
for i in range(3):
    print("-------- \n ",type(X_batch[i]), len(X_batch[i])," \n--------\n", X_batch[i]) # <class 'numpy.ndarray'>

'''
#########################
### contruction phase ###
#########################
tf.reset_default_graph()
n_neurons = 250
n_steps = 20
n_input = 1
n_output = 1
learning_rate = 0.001#0.0005 # # 0.02 # 0.005

X = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_steps, n_output])

# define network, 1 layer now
#cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
#cell =tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, actication=tf.rnn.relu)
#wrapped_cell = tf.contrib.rnn.OutputProjectionWrapper(cell, ouput_size=n_output)
#wrapped_cell = tf.contrib.rnn.OutputProjectionWrapper(
#    tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
#        output_size=n_output)
#outputs, states = tf.nn.dynamic_rnn(wrapped_cell, X, dtype=tf.float32)
# without using OutputProjectionWrapper
#basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.relu)
# 3 layers' lstm
#multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*3)
#multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*2)
#rnn_outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
# Add Dropout
keep_prob = 0.75
is_training = False 
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

##############################
##### Training Phase #########
##############################
n_iterations = 2500
batch_size = 50

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for iter in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iter % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iter,elapsed(time.time()-start_time), "MSE: ", mse)
    # evaluate
    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_input)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})
    #print(y_pred)
    #y_target = time_series(np.array(t_instance[1:].reshape(-1, n_steps, n_input)))
    #print(y_target)

plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "y*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.show()
'''
