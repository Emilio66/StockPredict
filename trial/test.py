# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

'''
'000001.SH', # 上证综指
'399001.SZ', # 深证成指
'399006.SZ', # 创业板指
'000300.SH', # 沪深 300
'000016.SH', # 上证 50
'000905.SH', # 中证 500
'''
# data_file = "G:\code\StockPredict\\data\\dataset\\close_2016-2017.csv"
data_file = "G:\code\StockPredict\\data\\dataset\\close_2012-2017.csv"
dataset = pd.read_csv(data_file,index_col=0, sep=',',names=['time','000001.SH','399001.SZ','399006.SZ','000300.SH','000016.SH','000905.SH'], skiprows=1,parse_dates=True)

# Fundamental Analysis
print(dataset.describe())

#plot
print("TYPE : ", type(dataset))
dataset.plot(lw=2.0)
plt.show()
'''
print("TYPE "," \t LENGTH")
print(type(dataset['close']),len(dataset['close']))
print(type(dataset), " \n --- \n",dataset.index, " \n --- \n",dataset.columns)
print(" subset: ",len(dataset.index[:-2]), dataset.index[:-2])

TIME_SPAN = 20
BATCH_SIZE = 3
n_input = 1
def next_batch(n_batch, n_step, index=0):
    X_batch, y_batch = [],[]
    for i in range(index, n_batch+index):
        X_batch.append(dataset['close'].values[i : i+n_step])
        y_batch.append(dataset['close'].values[i+1 : i+1+n_step])
    return np.array(X_batch).reshape(n_batch,n_step,n_input), np.array(y_batch).reshape(n_batch,n_step,n_input)

TRAIN_RATIO = 0.9
data_size = len(dataset['close']) // BATCH_SIZE
n_training = int(data_size * TRAIN_RATIO)
index = 0
for i in range(n_training):
    index = i*BATCH_SIZE
    X_batch, y_batch = next_batch(BATCH_SIZE, TIME_SPAN, index)
print(n_training, index)
'''