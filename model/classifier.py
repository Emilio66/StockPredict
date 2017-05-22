# -*- coding: UTF-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
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


# Batch Generator. ps: y batch is mmt classification
def next_batch(n_batch, n_step,n_input, index=0):
    X_batch, y_batch = [],[]
    # retrieve chunks in continuous way. every n_step as a chunk, not overlapped
    for i in range(index, n_batch*n_step+index,n_step):
        X_batch.append(dataset['close'].values[i : i+n_step])
        y_batch.append(dataset['label'].values[i+1+n_step]) #n time step share 1 label
    return np.array(X_batch).reshape(n_batch,n_step,n_input), np.array(y_batch)#.reshape(-1)


# how long will a span cover, e.g. 20 days (4 tradable weeks)
TIME_SPAN = 10
TRAIN_RATIO = 0.9#0.8
BATCH_SIZE = 3 


# testing data correctness
for i in range(1):
    X_batch, y_batch = next_batch(BATCH_SIZE, TIME_SPAN, 1, i*TIME_SPAN*BATCH_SIZE)

print("X_batch type:",type(X_batch)) #  <class 'list'>
print("-------- X_batch_shape ",X_batch.shape) # <class 'numpy.ndarray'>

#########################
### contruction phase ###
#########################
tf.reset_default_graph()
n_neurons = 150 #250
n_steps = TIME_SPAN
n_input = 1
n_output = 5 #5 class
n_layers = 3 #5#10 #5 #3
learning_rate = 0.001#0.0005 # # 0.02 # 0.005

X = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.int32, [None]) # one sequence one label

with tf.variable_scope("rnn", initializer=tf.contrib.layers.variance_scaling_initializer()):
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
states = states[0] #only need cell's states, omit hidden state
print("STATES: ", states)
states = tf.concat(axis=1, values=states) #sum up all neuron's result at final step
classifier= fully_connected(states, n_output, activation_fn=None)
print("Shape of Outputs: ",rnn_outputs, "shape of states: ", states)
print("Shape of Outputs: ",rnn_outputs.shape, "shape of states: ", states.shape)
print("Shape of classifier: ",classifier.shape, "Shape of y: ", y.shape)
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
n_epoch = 5#1#5#10
n_input = 1
n_batch = (len(dataset['close']) // TIME_SPAN) // BATCH_SIZE
n_training = int(n_batch * TRAIN_RATIO)
print("data size %d, train size %d" % (n_batch,n_training))

# add a test case //SHOULD BE MORE
size = len(dataset.index)
data = dataset['close'].values[size-TIME_SPAN-1:]
X_new = np.array(data[:-1])
y_new = np.array(data[1:])
X_test = X_new.reshape(-1,TIME_SPAN,n_input)
label = []
label.append(dataset['label'].values[-1])
y_test = np.array(label)#.reshape(-1,TIME_SPAN)
#print("X_test: ",X_test,"y_test",y_test)
date_axis = dataset.index[size-TIME_SPAN-1:]
#print("date list:",date_axis)
start_time = time.time()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for j in range(n_epoch):
        best_train_acc, sum_train_acc = 0., 0.
        for i in range(n_training):
            X_batch, y_batch = next_batch(BATCH_SIZE, TIME_SPAN,n_input, i*BATCH_SIZE)
            X_batch = X_batch.reshape((-1, n_steps, n_input))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if i % 3 == 0:
                acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                print(i,elapsed(time.time()-start_time), "Train accuracy: ", acc_train)
                if acc_train > best_train_acc:
                    best_train_acc = acc_train
                sum_train_acc += acc_train
        # evaluate
        print("       ---- TESTING ---")
        best_test_acc, sum_test_acc = 0., 0.
        for i in range(n_training,n_batch):
            X_test_batch, y_test_batch = next_batch(BATCH_SIZE, TIME_SPAN, n_input, i*BATCH_SIZE)
            X_test_batch = X_test_batch.reshape((-1, n_steps, n_input))
            sess.run(training_op, feed_dict={X: X_test_batch, y: y_test_batch})
            acc_test = accuracy.eval(feed_dict={X: X_test_batch, y: y_test_batch})
            if acc_test > best_test_acc:
                best_test_acc = acc_test
            sum_test_acc += acc_test
            print(i,elapsed(time.time()-start_time), "Test accuracy:", acc_test)
        print("================= Epoch ", j)
        print("   BEST Train Accuracy:", best_train_acc, " AVERAGE Train Accuracy:", sum_train_acc/n_training*3.0)
        print("   BEST Test Accuracy:", best_test_acc, " AVERAGE Test Accuracy:", sum_test_acc/(n_batch - n_training))
    
    # predict 1 value for ploting purpose
    res = sess.run(classifier, feed_dict={X: X_test})
    _, y_pred = tf.nn.top_k(res)  # return the indices
    y_pred = y_pred.eval() 
    print("Class: ",y_pred)

# plotting
y_predict_series = []
print(type(y_pred), y_pred.shape, len(y_pred))
y_pred = y_pred[0,:]
categoryList = [-0.04, -0.01, 0, 0.01, 0.04] # rough category value: indices represents category
for i in range(len(y_pred)):
    try:
        y_predict_series.append(X_new[i] * categoryList[y_pred[i]] + X_new[i])
    except IndexError as e:
        print ("index error",y_pred[i],e)
plt.title("Testing the model", fontsize=14)
plt.plot(date_axis[:-1], X_new, "bo", markersize=10, label="instance")
plt.plot(date_axis[1:], y_new, "y*", markersize=10, label="target")
plt.plot(date_axis[-1], y_predict_series, "r.", markersize=14, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.show()
