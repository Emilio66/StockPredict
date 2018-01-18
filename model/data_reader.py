import numpy as np
import pandas as pd
import time
import math

'''
Reading Macro economy factors from CSV
Notice: data are distributed monthly, need to be filled for daily usage
'''
def strip_comma(x):
    return float(str(x).replace(',',''))
def read_macro_economy(base_dir = '../data/macro_economy/', 
                       filename = 'china_macro_economy_daily.csv',
                       start_date = '2002-01-04', end_date = '2017-11-30',
                       names = [i for i in range(31)],
                       usecols = None):
    filename = base_dir +'/'+ filename
    print (filename) #中文读取出问题，所以skip row1
    df = pd.read_csv(filename, index_col=0, sep=',', 
                     skiprows=1, usecols=usecols,
                     names = names, parse_dates=True,
                     converters = {11: strip_comma, 22: strip_comma}
                    )
    return df[start_date : end_date]


'''
Reading World economy factors sponsored by OECD from CSV
Notice: data are distributed monthly, need to be filled for daily usage
'''
def read_world_economy(base_dir = '../data/macro_economy/', 
                       filename = 'OECD-world-economy-daily.csv',
                       start_date = '2002-01-04', end_date = '2017-11-30',
                       names = [i for i in range(46)],
                       usecols = None):
    filename = base_dir +'/'+ filename
    print (filename) 
    df = pd.read_csv(filename, index_col=0, 
                     skiprows=1, usecols=usecols,parse_dates=True,
                     names = names
                    )
    return df[start_date : end_date]

'''
Reading Top10 Components CSV
Data has been assigned weight according to their ratio in the market

# Ref: data calculated from data/generate/FetchingComponentsData.ipynb
'''
def read_components(base_dir = '../data/components/', 
                       filename = 'components-top10.csv',
                       start_date = '2002-01-04', end_date = '2017-11-30',
                       names = [i for i in range(10)]):
    filename = base_dir +'/'+ filename
    print (filename) 
    df = pd.read_csv(filename, index_col=0, 
                     skiprows=1,parse_dates=True,
                     names = names
                    )
    df = df.fillna(0)
    return df[start_date : end_date]

'''
Reading ohlcv transaction data for a stock

'''
def readWSDFile(baseDir, stockCode, startDate='2005-01-04', endDate= '2015-12-31', usecols=None, 
                names=['date','pre_close','open','high','low','close','change','chg_range',
                                               'volume','amount','turn']):
    # 解析日期
    filename = baseDir+stockCode+'/'+stockCode+'.csv'
    print (filename, "===============")
    dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d').date()
    df = pd.read_csv(filename, index_col=0, sep=',', header=None,usecols=usecols,
                            skiprows=1, names=names,
                           parse_dates=True, date_parser=dateparse)
    df = df.fillna(0)
    return df[startDate : endDate]

'''
Reading Technical indicators of a stock
'''
def readWSDIndexFile(baseDir, stockCode, startYear, yearNum=1):
    # parse date
    dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d').date()

    df = 0
    for i in range(yearNum):
        tempDF = pd.read_csv(baseDir+'I'+stockCode+'/wsd_'+stockCode+'_'+str(startYear+i)+'.csv', index_col=0, sep=',', parse_dates=True, date_parser=dateparse
                             # , usecols=usecols
                             )
        if i==0: df = tempDF
        else: df = df.append(tempDF)
    df = df.fillna(0)
    return df