from keras.models import Sequential
from keras.layers import LSTM, Dense

import numpy as np

'''
Preparing data
'''
import pandas as pd
import time
import matplotlib.pyplot as plt
import math

'''
读入一支股票指定年份的ohlcv数据
输入:baseDir,stockCode为字符, startYear,yearNum为整数，
输出:dataframe
'''
def readWSDFile(baseDir, stockCode, startYear, yearNum=1, usecols=None, 
                names=['date','pre_close','open','high','low','close','change','chg_range',
                                               'volume','amount','turn']):
    # 解析日期
    filename = baseDir+stockCode+'/'+stockCode+'.csv'
    print (filename, "===============")
    dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d').date()
    df = pd.read_csv(filename, index_col=0, sep=',', header=None,usecols=usecols,
                            skiprows=1, names=names,
                           parse_dates=True, date_parser=dateparse)
    return df['2005-01-04':'2015-12-31']

'''
读入一支股票指定年份的技术指标
输入:baseDir,stockCode为字符, startYear,yearNum为整数，
输出:dataframe
'''
def readWSDIndexFile(baseDir, stockCode, startYear, yearNum=1):
    # 解析日期
    dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d').date()

    df = 0
    for i in range(yearNum):
        tempDF = pd.read_csv(baseDir+'I'+stockCode+'/wsd_'+stockCode+'_'+str(startYear+i)+'.csv', index_col=0, sep=',', parse_dates=True, date_parser=dateparse
                             # , usecols=usecols
                             )
        if i==0: df = tempDF
        else: df = df.append(tempDF)
    return df

# 整理好多因子输入，以dataframe返回数据+标签
from sklearn import preprocessing
def data_prepare(retrace = 0.618):
    # prepare data
    baseDir = '../data/'
    stockCodes = ['000300.SH']
    i = 0
    startYear = 2005
    number =11
    usecols = None#[0,5,6]
    names = ['date','close','change']
    df = readWSDFile(baseDir, stockCodes[i], startYear, number, usecols)
    #dfi = readWSDIndexFile(baseDir, stockCodes[i], startYear, number)
    allDF = df#pd.concat([df, dfi], axis=1)
    sample_num = np.shape(df)[0]
    labelDF = pd.Series(np.zeros(sample_num))
    print ("Factors Shape:", np.shape(df), np.shape(allDF))
    
    # 求出 trend
    price = df['close']
    start = 0
    while price[start] > price[start+1]:
        labelDF[start] = 1 #flat
        start +=1
    print("----- start: ",start)
    #find peak, find trough, calculate retracement and label trend accordingly
    i = start
    while i < sample_num - 1:
        cursor = i
        while cursor < sample_num - 1 and price[cursor] <= price[cursor+1]:
            cursor += 1
        peak = cursor
        while cursor < sample_num - 1 and price[cursor] >= price[cursor+1]:
            cursor += 1
        trough = cursor
        retracement = (price[peak] - price[trough]) / (price[peak] - price[i])
        mark = 1 # flat
        if retracement < retrace:
            mark = 2 # UP
        elif retracement > 1 + retrace:
            mark = 0 # DOWN
        for k in range(i, cursor+1):
            labelDF[k] = mark
        i = cursor

    print("---- Trend Distribution Check --------")
    print(labelDF.value_counts().sort_index())
    
    # make a deep copy of Price Difference before normalizing
    priceDF = allDF['change'].copy(deep=True)
    # scikit-learn normalize or: keras.utils.normalize(x)
    scaler = preprocessing.MinMaxScaler()
    input_data = scaler.fit_transform(allDF)
    print ("input data shape: ", np.shape(input_data)) #  days *  factors
    print ("input label shape: ", np.shape(labelDF))
   
    return input_data, labelDF, priceDF # train/test data, labels and prices for yield calucluation

###### Hyper paramters #########
time_steps = 8
n_neurons = 300
num_classes = 3
batch_size = 20 # specify batch size explicitly; no shuffle but successive sequence
n_epoch = 23
# get training data
train_ratio = 0.9
#_, dataset, _ = data_prepare()
dataset,labels, _ = data_prepare()
segment_num = (len(dataset) - time_steps - 1) // batch_size # rollingly use data
train_size = int(segment_num * train_ratio)
test_size = segment_num - train_size
data_dim = 1#np.shape(dataset)[1] #input

#divide training/validation dataset
#train_x = dataset[0 : train_size * batch_size + time_steps]
#test_x = dataset[train_size * batch_size : (train_size + test_size) * batch_size + time_steps]

train_x = labels.iloc[0 : train_size * batch_size + time_steps]
test_x = labels.iloc[train_size * batch_size : (train_size + test_size) * batch_size + time_steps]
#label is just 1 step further after sequence data
train_y = labels.iloc[time_steps : train_size * batch_size + time_steps]
test_y = labels.iloc[train_size * batch_size + time_steps: (train_size + test_size) * batch_size + time_steps]

# construct training data, 1 step forward, keep rolling
train_x = np.array(train_x)
train_sample = len(train_x) - time_steps
b = np.array([[]])
for i in range(train_sample):
    b = np.append(b, train_x[i : time_steps + i])
train_x = b.reshape(train_sample, time_steps, data_dim)
print("training size: ", train_sample)

test_x = np.array(test_x)
test_sample = len(test_x) - time_steps
b = np.array([[]])
for i in range(test_sample):
    b = np.append(b, test_x[i : time_steps + i])
test_x = b.reshape(test_sample, time_steps, data_dim)
print("testing size: ", test_sample)

train_y = np.array(train_y, dtype=np.int32)
test_y = np.array(test_y, dtype=np.int32)


# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(n_neurons, return_sequences=True,stateful=True,
               batch_input_shape=(batch_size, time_steps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(n_neurons, return_sequences=True, stateful=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(n_neurons, return_sequences=True, stateful=True)) 
model.add(LSTM(n_neurons, stateful=True))  # return a single vector of dimension 32
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', #for integer class, not one hot encoding
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(train_x, train_y,
          batch_size=batch_size, 
          epochs=n_epoch,
          shuffle=False,
          validation_data=(test_x, test_y))
print ("Params: ", "time_steps:", time_steps, "  n_neurons:", n_neurons, " n_epoch: ", n_epoch)
