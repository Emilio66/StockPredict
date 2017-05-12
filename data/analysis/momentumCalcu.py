import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

filename = "/../dataset/close_2016-2017.csv"
file_path = os.path.join(os.path.dirname(__file__) + filename)
dataset = pd.read_csv(file_path,index_col=0, sep=',', usecols=[0,6], skiprows=1, names=['date','close'],parse_dates=True)

# Fundamental Analysis
print("-------------------- Describe Dataset ------------------")
print(dataset.describe())

# calculate momentum: Mt = (CLOSE(t) -CLOSE(t-1))/CLOSE(t)
dataset['mmt'] = 0.0
for i in range(1, len(dataset)):
    dataset['mmt'][i] = (dataset['close'][i] - dataset['close'][i-1]) / dataset['close'][i]

# count (histogramming)
print("-------------------- Momentum Distribution ------------------")
print(dataset['mmt'].value_counts(bins=5).sort_index())

pd.set_option('mode.chained_assignment',None)
# classify by counts
dataset['label'] = 0
# read only
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
print(dataset['label'].value_counts().sort_index())
# simple way
# dataset.loc[:,'label'][mmt_series < -0.02] = 0
# dataset.loc[:,'label'][mmt_series > 0.02] = 4
# dataset.loc[:,'label'][(mmt_series >= -0.02) & (mmt_series < -0.005)] = 1
# dataset.loc[:,'label'][(mmt_series >= -0.005) & (mmt_series <= 0.005)] = 2
# dataset.loc[:,'label'][(mmt_series > 0.005) & (mmt_series <= 0.02)] = 3

# EXCEPTION WARNNING: SettingWithCopyWarning (Pandas don't know whether a copy or view return, no guarantee of successful write)
# print(dataset)
# dataset['label'].plot(lw = 2.)
# plt.show()

# # dataset['mmt'].value_counts(bins=10, ascending=False).plot(lw = 2.)
# plt.hist(dataset['mmt'].values, bins = 5)
# plt.xlabel('Momentum')
# plt.ylabel('Counts')
# # plt.grid(True)
# plt.show()