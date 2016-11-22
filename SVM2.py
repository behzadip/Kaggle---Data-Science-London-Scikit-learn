# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:24:49 2016

@author: behzad
"""

import numpy as np
from sklearn import svm
from sklearn import decomposition
from sklearn import cross_validation as cv
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import preprocessing as pp
from sklearn.neighbors import KNeighborsClassifier

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#reading data
train = np.loadtxt(open("train.csv", "rb"), delimiter=",", skiprows=0)
trainLabels = np.loadtxt(open("trainLabels.csv", "rb"), delimiter=",", skiprows=0)
test = np.loadtxt(open("test.csv", "rb"), delimiter=",", skiprows=0)

#norm = pp.Normalizer()
#train_norm = norm.fit_transform(train)
#test_norm = norm.transform(test)

#start a stratified k fold cross validation
cvk = cv.StratifiedKFold(trainLabels, n_folds=5)

# using RFECV to get an idea of number of important features
from sklearn.feature_selection import RFECV
est = svm.SVC(kernel='linear')
rfecv = RFECV(est,cv=cvk)
rfecv.fit(train, trainLabels)
#rfecv.fit(train_norm, trainLabels)
print("Optimal number of features : %d" % rfecv.n_features_)

#pca = decomposition.PCA(n_components=12, whiten=True)
#train_pca = pca.fit_transform(train)
#test_pca = pca.transform(test)

#looking at PCA, around 98% of variability can be retained by only 12 variables out of 40
pca2 = decomposition.PCA(n_components=12, whiten=True)
XY = pca2.fit_transform(np.r_[train, test])
train_pca2 = XY[:1000,:]
test_pca2 = XY[1000:,:]

# define grid search parameters for SVM model
param_grid = {'C': 10.0 ** np.arange(6.5,7.5,.25), 'gamma': 10.0 ** np.arange(-1.5,0.5,.25), 'kernel': ['rbf']}
grid = GridSearchCV(svm.SVC(), param_grid, cv=cvk)
grid.fit(train_pca2, trainLabels)
print grid.best_score_
#predict on test set
pred = grid.predict(test_pca2).astype(int)
#print metrics.classification_report(y_test.astype(np.int), pred)

idd = range(1, len(test)+1)
submit = zip(idd, pred)
np.savetxt("SVCSubmit2.csv", submit, header= 'id,Solution', delimiter=",", fmt='%d', comments='')

# trying a k neighbor classifier
kclass = KNeighborsClassifier(n_neighbors=5,algorithm='auto',weights='distance')

print cross_val_score(kclass, train_pca2, trainLabels, cv=cvk)
kclass.fit(train_pca2, trainLabels)
# predicting on test set
res = kclass.predict(test_pca2).astype(int)


#-----------------spliting train set into two train sets 80% and 20%
train1, train2, label1, label2 = train_test_split(train_pca2,trainLabels,test_size=0.2)
cvk1 = cv.StratifiedKFold(label1, n_folds=5)
# train an SVM on 80% of data and predict for 20%
grid = GridSearchCV(svm.SVC(), param_grid, cv=cvk1)
grid.fit(train1, label1)
pred1 = grid.predict(train2).astype(int)

# train a k neighbor model on 80% of data and predict for 20%
kclass.fit(train1, label1)
pred2 = kclass.predict(train2).astype(int)

cvk2 = cv.StratifiedKFold(label2, n_folds=3)
train3 = np.vstack((train2.T,np.vstack((pred1,pred2)))).T
# train an SVM model to use 12 pca features and 2 prediction of above models
final = svm.SVC()
final.fit(train3, label2)
print cross_val_score(final, train3, label2, cv=cvk2)
# predict on test set
test_all = np.vstack((test_pca2.T,np.vstack((pred,res)))).T
final_pred = final.predict(test_all).astype(int)
# writing the submission file
idd = range(1, len(test)+1)
submit = zip(idd, final_pred)
np.savetxt("SVCSubmit5.csv", submit, header= 'id,Solution', delimiter=",", fmt='%d', comments='')