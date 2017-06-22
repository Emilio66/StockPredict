from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import sys
sys.path.append("..")
from data.dataset.BatchGenerator import *

# previous 30 days predictin the next 10 days
HISTORY_DAYS = 4
PREDICT_DAYS = 1
# 85% for training
TRAIN_PER = 0.85

features, labels = generateSVMData('../data/dataset/close_weekly-2007-2017.csv', HISTORY_DAYS, PREDICT_DAYS)

# C, gamma search space
tuned_parameters = {'C': np.logspace(start=-10, stop=10, num=21, base=2), 'gamma': np.logspace(start=-10, stop=10, num=21, base=2)}

train_num = int(len(labels) * TRAIN_PER)

# Grid Search & cross-validation
clf1 = GridSearchCV(svm.SVC(), tuned_parameters, cv=[(list(range(train_num)), list(range(train_num, len(labels))))])
clf1.fit(features, labels)

#print(clf1.cv_results_)
print(clf1.best_params_)
print(clf1.best_score_)

#clf1.predict(test_features, test_labels)
