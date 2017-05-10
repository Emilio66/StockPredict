# -*- coding: UTF-8 -*-
# pylint: disable=I0011,C0111, C0103,C0326,C0301, C0304, W0401,W0614
from cassandra.cluster import Cluster
from cassandra.util import Date
import time
import datetime
import math
import numpy as np
import csv
import os

#####################################################################################
## Generate training file within required periods in CSV file separated by '\t' #####
#####################################################################################
def generateData(fileName, startTime, endTime, stocks, table = "factors_day", TYPE='D'):
    if startTime > endTime:
        return

    cluster = Cluster(['192.168.1.111'])
    session = cluster.connect('factors') #connect to the keyspace 'factors'

    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Retrieving data: ", len(stocks))
    #time list
    rows = session.execute('''
        select * from transaction_time 
        where type= %s and time > %s and time < %s ALLOW FILTERING;''', [TYPE,startTime, endTime])

    SQL = "SELECT value FROM "+table+" WHERE stock = ? AND factor = 'close' and time >= '" + str(startTime) +"' and time <= '" + str(endTime)+"'"
    preparedStmt = session.prepare(SQL)

    dateList = []
    for row in rows:
        dateList.append(row.time)
    # 拉取数据,一次拉一只股票
    dataList = []
    for stock in stocks:
        rows = session.execute(preparedStmt,(stock,))
        data = []
        for row in rows:
            data.append(row[0])
        dataList.append(data)
    cluster.shutdown()

    # 数据写入文件中
    f = open(fileName, "w")
    f.write('time')
    for stock in stocks:
        f.write(','+stock)
    f.write('\n')
    colNum = len(stocks)
    rowNum = len(dateList)
    for i in range(rowNum):
        f.write(str(dateList[i]))
        for s in range(colNum):
            try:
                data = dataList[s][i]
                if math.isnan(data):
                    data = 0  # default value
                f.write(',' + str(data))
            except IndexError:
                print ("End of reading and writing daily close data...")
                f.close()
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Writing to ', fileName, ' complete!')
                return
            #print (timeList[i],stocks[s],dataList[s][0][i])
        f.write('\n')
    f.close()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Writing to ',fileName,' complete!')

##############################################
################# EXAMPLE USAGE ##############
stock_indexes=["000001.SH","399001.SZ",'399006.SZ','000300.SH','000016.SH','000905.SH']
generateData("E:\\close_2012-2017.csv",datetime.date(2012,1,1),datetime.date(2017,4,30), stock_indexes)