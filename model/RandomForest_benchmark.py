from sklearn.ensemble import RandomForestClassifier
import sys
import time
sys.path.append("..")
from data.dataset.BatchGenerator import *
HISTORY_DAYS = 10
PREDICT_DAYS = 1

filename = './exp/final_RandomForest_datasets2.txt'
lines = []
for hasRetrace in range(2):
    for col in [1,2,4,5,6]:
        trainX, trainY, testX, testY = generateEnsembleData('../data/dataset/close_2007-2017.csv',
         HISTORY_DAYS, PREDICT_DAYS, no_retrace=hasRetrace, use_weight=0, col=col)

        minE = 3
        maxE = 60
        step = 3
        bestResult = 0
        sumScore = 0
        cnt = 0
        for i in range(minE, maxE, step):
            clf = RandomForestClassifier(n_estimators=i)
            clf = clf.fit(trainX, trainY)
            score = clf.score(testX,testY)
            sumScore += score
            cnt += 1
            if score > bestResult:
                bestResult = score
                print('best: %f @ %d' % (bestResult, i))
        log = 'Dataset: '+ str(col) +' NoRetrace: ' +\
            str(bool(hasRetrace)) + ' Avg: ' + str(sumScore/cnt) +' Best: '+str(bestResult)+ '\n'
        lines.append(log)
        print(log)
with open(filename, 'w') as f:
    f.writelines(lines)
print("Writing file", filename, "complete!")
