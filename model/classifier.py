# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

######################################
####### Data Preparation #############
######################################
data_file = "../data/dataset/close_2016-2017.csv"
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
print("Total: ",len(dataset['label']),dataset['label'].value_counts().sort_index())
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , " ------ Complete Data Preparation")


# Batch Generator. ps: y batch is mmt classification
def next_batch(n_batch, n_step,n_input, index=0):
    X_batch, y_batch = [],[]
    for i in range(index, n_batch+index):
        X_batch.append(dataset['close'].values[i : i+n_step])
        y_batch.append(dataset['label'].values[i : i+n_step])
    return np.array(X_batch).reshape(n_batch,n_step,n_input), np.array(y_batch).reshape(n_batch,n_step,n_input)


# how long will a span cover, e.g. 20 days (4 tradable weeks)
TIME_SPAN = 20
TRAIN_RATIO = 0.9#0.8
BATCH_SIZE = 3

# testing data correctness
for i in range(1):
    X_batch, y_batch = next_batch(BATCH_SIZE, TIME_SPAN, 1, i*TIME_SPAN*BATCH_SIZE)

print(type(X_batch)) #  <class 'list'>
print("-------- \n ",X_batch.shape," \n--------\n", X_batch) # <class 'numpy.ndarray'>

#########################
### contruction phase ###
#########################
tf.reset_default_graph()
n_neurons = 250
n_steps = TIME_SPAN
n_input = 1
n_output = 1
n_layers = 3
learning_rate = 0.001#0.0005 # # 0.02 # 0.005

X = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_steps, n_output])

# define network, 1 layer now
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.relu)
cells = lstm_cell
# 3 layers' lstm
#cells = tf.contrib.rnn.MultiRNNCell([lstm_cell]*3)

# Add Dropout
is_training = False 
keep_prob = 0.75
if is_training:
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cells, input_keep_prob=keep_prob)

rnn_outputs, states = tf.nn.dynamic_rnn(cells, X, dtype=tf.float32)
classifier= fully_connected(states, n_outputs, activation_fn=None)
print("Shape of Outputs: ",outputs.shape, "shape of states: ", states.shape)
print("Shape of classifier: ",classifier.shape)
# stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
# stacked_outputs = fully_connected(stacked_rnn_outputs, n_output, activation_fn=None)
# outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_output])

# softmax + cross entropy calculation
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=classifier)
# define loss function & optimize method
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
# measurement
correct = tf.nn.in_top_k(classifier, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

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
n_input = 1
data_size = len(dataset['close']) // BATCH_SIZE
n_training = int(data_size * TRAIN_RATIO)
print("data size %d, train size %d" % (data_size,n_training))

# add a test case //SHOULD BE MORE
size = len(dataset.index)
data = dataset['close'].values[size-TIME_SPAN-1:]
X_new = np.array(data[:-1])
y_new = np.array(data[1:])
X_test = X_new.reshape(-1,TIME_SPAN,n_input)
y_test = y_new.reshape(-1,TIME_SPAN,n_input)
print("X_test: ",X_test,"y_test",y_test)
date_axis = dataset.index[size-TIME_SPAN-1:]
print("date list:",date_axis)
start_time = time.time()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for j in range(n_epoch):
        for i in range(n_training):
            X_batch, y_batch = next_batch(BATCH_SIZE, TIME_SPAN,n_input, i*BATCH_SIZE)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if i % 10 == 0:
                acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                acc_test = accuracy.eval(feed_dict={X: X_test,y: y_test})
                print(i,elapsed(time.time()-start_time), "Train accuracy: ", acc_train, "Test accuracy:", acc_test)
        # evaluate
        print("---------Epoch ", j, " Train accuracy:", acc_train, " Test accuracy:", acc_test)
        # X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_input)))
        y_pred = sess.run(outputs, feed_dict={X: X_test})
        #print(y_pred[0,:-1,0],'\n',X_test[0,1:,0])
    #print(y_pred)
    #y_target = time_series(np.array(t_instance[1:].reshape(-1, n_steps, n_input)))
    #print(y_target)
y_predict_series = []
y_pred = y_pred[0,:,0]
categoryList = [-0.04, -0.01, 0, 0.01, 0.04] # rough category value
for i in range(len(y_pred)):
    y_predict_series.append(X_new[i] * categoryList[y_pred[i]] + X_new[i])
plt.title("Testing the model", fontsize=14)
plt.plot(date_axis[:-1], X_new, "bo", markersize=10, label="instance")
plt.plot(date_axis[1:], y_new, "y*", markersize=10, label="target")
plt.plot(date_axis[1:], y_predict_series, "r.", markersize=14, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.show()
