from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import sys
import time
sys.path.append("..")
from data.dataset.BatchGenerator import *

HISTORY_DAYS = 10
PREDICT_DAYS = 1
TRAIN_PER = 0.9
lines = []
minR, maxR = 0, 8

for i in range(minR, maxR):
	features, labels = generateSVMData('../data/dataset/close_weekly-2007-2017.csv', HISTORY_DAYS, PREDICT_DAYS, no_retrace=False, use_weight=i)
	
	# C, gamma search space
	tuned_parameters = {'C': np.logspace(start=-10, stop=10, num=21, base=2), 'gamma': np.logspace(start=-10, stop=10, num=21, base=2)}

	train_num = int(len(labels) * TRAIN_PER)

	# Grid Search & cross-validation
	clf1 = GridSearchCV(svm.SVC(), tuned_parameters, cv=[(list(range(train_num)), list(range(train_num, len(labels))))])
	clf1.fit(features, labels)

	#print(clf1.cv_results_)
	best_param = clf1.best_params_
	best_score = clf1.best_score_
	print(best_param)
	print(best_score)
	line = 'weight' + str(i) + '\t' +str(best_score) +'\t' +str(best_param) +'\n'
	lines.append(line)
	
filename="./exp/svm"+str(minR)+'-'+str(maxR)+"-weighted-func0-7.txt"
with open(filename, 'w') as f:
	f.writelines(lines)
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) , "Writing file", filename, "complete!")

#clf1.predict(test_features, test_labels)
