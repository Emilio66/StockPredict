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
trace = 0.618
price = dataset['close']
size = len(price)
# for i in range(2, size, interval):
# 	#print(i)
# 	if i >= size - interval:
# 		break
# 	trend = 0
# 	diff = [0 for i in range(interval)]
# 	for j in range(interval):
# 		diff[j] = price[i+j+1] - price[i+j]
# 	# P1 - T1 < (P1 - T0) * trace	
# 	if price[i+1]+price[i+2]+price[i+3]-price[i-1]-price[i-2]-price[i] > 0 and (price[i+1]+price[i+2]+price[i+3]-price[i-1]-price[i-2]-price[i])*(1+trace) <  (price[i+4]+price[i+5]+price[i+6]-price[i+1]-price[i+2]-price[i+3]):
# 		trend = 1
# 	elif price[i+1]+price[i+2]+price[i+3]-price[i-1]-price[i-2]-price[i] < 0 and -(price[i+1]+price[i+2]+price[i+3]-price[i-1]-price[i-2]-price[i])*(1+trace) <  -(price[i+4]+price[i+5]+price[i+6]-price[i+1]-price[i+2]-price[i+3]):
# 		trend = -1
# 	if price[i+1] < price[i+3] and price[i+3] < price[i+5] and diff[1] < -diff[0]*trace and diff[3] < -diff[2]*trace and diff[5] < -diff[4]*trace:
# 		trend = 1
# 		print ('UP')
# 	# P1 - T1 > (P1 - T0) * trace
# 	elif price[i+1] > price[i+3] and price[i+3] > price[i+5] and diff[1] > -diff[0]*trace and diff[3] > -diff[2]*trace and diff[5] > -diff[4]*trace:
# 		trend = -1
# 		print ('DOWN')
# 	mark = 2000
# 	if trend < 0:
# 		mark = 1500
# 	elif trend > 0:
# 		mark = 2500
# 	for k in range(i, i+interval):
# 		dataset['trend'][k] = mark

# initialize with a low point
start = 0
while price[start] > price[start+1]:
	start +=1
print("start: ",start)
#find peak, find trough, calculate retracement and label trend accordingly
i = start
while i < size - 1:
	cursor = i
	while cursor < size - 1 and price[cursor] < price[cursor+1]:
		cursor += 1
	peak = cursor
	while cursor < size - 1 and price[cursor] > price[cursor+1]:
		cursor += 1
	trough = cursor
	retracement = (price[peak] - price[trough]) / (price[peak] - price[i])
	mark = 2000 # flat
	if retracement < trace:
		mark = 2500 # UP
	elif retracement > 1 + trace:
		mark = 1500 # DOWN
	for k in range(i, cursor+1):
		dataset['trend'][k] = mark
	i = cursor
#plot
print("TYPE : ", type(dataset))
dataset.plot(lw=2.0)
plt.show()
