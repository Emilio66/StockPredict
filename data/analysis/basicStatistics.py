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
#data_file = "../dataset/close_2002-2017.csv"
data_file = "../dataset/close_weekly-2007-2017.csv"
#data_file = "../dataset/close_weekly-2002-2017.csv"
#dataset = pd.read_csv(data_file,index_col=0, sep=',',names=['time','000001.SH','399001.SZ','399006.SZ','000300.SH','000016.SH','000905.SH'], skiprows=1,parse_dates=True)
dataset = pd.read_csv(data_file,index_col=0, sep=',', usecols=[0,5],names=['time','close'], skiprows=1,parse_dates=True)
# Fundamental Analysis
print(dataset.describe())

#TRENDS COMPUTE
dataset['trend']=0.0
interval = 6
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
trace = 0.618
while price[start] > price[start+1]:
	start +=1
print("start: ",start)
#find peak, find trough, calculate retracement and label trend accordingly
i = start
waves = []
retraces = []
while i < size - 1:
	cursor = i
	while cursor < size - 1 and price[cursor] <= price[cursor+1]:
		cursor += 1
	peak = cursor
	while cursor < size - 1 and price[cursor] >= price[cursor+1]:
		cursor += 1
	trough = cursor
	retracement = (price[peak] - price[trough]) / (price[peak] - price[i])
	if retracement < 5:
		retraces.append(retracement)
	mark = 1 # flat
	if retracement < trace:
		mark = 2 # UP
	elif retracement > 1 + trace:
		mark = 0 # DOWN
	for k in range(i, cursor+1):
		dataset['trend'][k] = mark
	waves.append(mark)
	i = cursor

print(dataset['trend'].value_counts().sort_index())
retraces = pd.Series(retraces)
print(retraces.value_counts(bins = 10).sort_index())
print(retraces.describe())

dataset['mmt'] = 0.0
for i in range(1, len(dataset)):
    dataset['mmt'][i] = (dataset['close'][i] - dataset['close'][i-1]) / dataset['close'][i-1]

pd.set_option('mode.chained_assignment',None)
    # classify by counts
dataset['label'] = 0
mmt_series = dataset['mmt']
for i in range(len(dataset)):
    mmt = mmt_series[i]
    if mmt < -0.01: 
        dataset['label'][i] = 0 #down
    elif mmt <= 0.01:
        dataset['label'][i] = 1 #flat
    else:
        dataset['label'][i] = 2 #up

#plot
#print("TYPE : ", type(dataset))
dataset.plot(y=['close', 'trend', 'label'],lw=2.0, subplots=True)
plt.show()

#plt.plot(waves, markersize=12, linewidth=2, label="Waves")
#plt.show()
## search for the optimal combination
## wave theory: 2 up + 1 no + 1down or 2 downs
'''
print(dataset['trend'].value_counts().sort_index())
up_max = 10
down_max = 10
trends = waves
pattern1 = [2,2,2,0]
pattern2 = [2,2,2,1]
pattern3 = [2,2,1,0]
pattern4 = [2,2,1,1]
size = len(waves)
cnt1, cnt2, cnt3, cnt4, others = 0,0,0,0,0
for i in range(size - 4):
	wave = waves[i : i + 4]
	if wave == pattern1:
		cnt1 += 1
	elif wave == pattern2:
		cnt2 += 1
	elif wave == pattern3:
		cnt3 += 1
	elif wave == pattern4:
		cnt4 += 1
	else:
		others +=1
print(cnt1, cnt2, cnt3, cnt4, others, size)

for i in range(down_max):
	for j in range(1, up_max):
		combo = 0
		pattern = [ 2 for x in range(j)]
		n = 0
		while n < i:
			pattern.append(0)
			n += 1
		k = 0
		while k < len(trends) - i - j:
			is_fit = (trends[k : k+j] == np.array(pattern))
			for l in range(k, k+j):
				if l > len(trends) - j -1:
					break;
				if trends[l] != 2:
					is_fit = False
					break;
			
			if is_fit is True:
				for m in range(k+j, k+j+i):
					if m > len(trends) - 1:
						break;
					if trends[m] != 0:
						is_fit = False
						break;
			if is_fit is False:
				k += 1
			else:
				combo += 1
				k += i+j
		print('UP/DOWN %d/%d total %d' % (j, i, combo))
'''

