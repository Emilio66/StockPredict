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
#data_file = "../dataset/close_2007-2017.csv"
data_file = "../dataset/close_weekly-2007-2017.csv"
#data_file = "../dataset/close_weekly-2002-2017.csv"
#dataset = pd.read_csv(data_file,index_col=0, sep=',',names=['time','000001.SH','399001.SZ','399006.SZ','000300.SH','000016.SH','000905.SH'], skiprows=1,parse_dates=True)
dataset = pd.read_csv(data_file,index_col=0, sep=',', usecols=[0,1],names=['time','close'], skiprows=1,parse_dates=True)
# Fundamental Analysis
print(dataset.describe())

#TRENDS COMPUTE
dataset['trend']=0.0
interval = 6
retrace = 0.5
price = dataset['close']
size = len(dataset)
for i in range(0, size, interval):
	if i >= size - interval:
		break
	trend = 0
	diff = [0 for i in range(interval)]
	for j in range(interval):
		diff[j] = price[i+j+1] - price[i+j]
	# P1 - T1 < (P1 - T0) * trace	
	if price[i+1] < price[i+3] and price[i+3] < price[i+5] and diff[1] < -diff[0]*trace and diff[3] < -diff[2]*trace and diff[5] < -diff[4]*trace:
		trend = 1
	# P1 - T1 > (P1 - T0) * trace
	elif price[i+1] > price[i+3] and price[i+3] > price[i+5] and diff[1] > -diff[0]*trace and diff[3] > -diff[2]*trace and diff[5] > -diff[4]*trace:
		trend = -1
	mark = 2000
	if trend < 0:
		mark = 1500
		print ('DOWN')
	elif trend > 0:
		mark = 2500
		print ('UP')
	for k in range(i, i+interval):
		dataset['trend'][k] = mark
#plot
print("TYPE : ", type(dataset))
dataset.plot(lw=2.0)
plt.show()
