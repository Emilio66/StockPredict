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
data_file = "../dataset/close_2007-2017.csv"
dataset = pd.read_csv(data_file,index_col=0, sep=',',names=['time','000001.SH','399001.SZ','399006.SZ','000300.SH','000016.SH','000905.SH'], skiprows=1,parse_dates=True)

# Fundamental Analysis
print(dataset.describe())

#plot
print("TYPE : ", type(dataset))
dataset.plot(lw=2.0)
plt.show()
