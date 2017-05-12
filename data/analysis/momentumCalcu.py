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

# classify by counts
dataset['label'] = 0

for i in range(len(dataset)):
    mmt = dataset['mmt'][i]
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

# EXCEPTION WARNNING: SettingWithCopyWarning (Pandas don't know whether a copy or view return, no guarantee of successful write)

dataset['label'].plot(lw = 2.)
plt.show()

# dataset['mmt'].value_counts(bins=10, ascending=False).plot(lw = 2.)
plt.hist(dataset['mmt'].values, bins = 5)
plt.xlabel('Momentum')
plt.ylabel('Counts')
# plt.grid(True)
plt.show()