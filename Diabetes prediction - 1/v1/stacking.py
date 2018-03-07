# -*- coding: utf-8 -*-
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

reader = pd.read_csv('../data/data_scale_train.csv',chunksize = 10000000) 
model = []       
for chunk in reader:
    df = pd.DataFrame(chunk)
    model.append(df)
X_train = pd.concat(model)

reader = pd.read_csv('../data/label.csv',chunksize = 10000000) 
model = []       
for chunk in reader:
    df = pd.DataFrame(chunk)
    model.append(df)
label = pd.concat(model)

reader = pd.read_csv('../data/data_scale_test.csv',chunksize = 10000000) 
model = []       
for chunk in reader:
    df = pd.DataFrame(chunk)
    model.append(df)
x_tes = pd.concat(model)

X_dev = X_train.values
Y_dev = label['target'].values

X_te = x_tes.values

skf = KFold(len(X_dev), n_folds = 5, shuffle=True, random_state=520)

clfs = [
GradientBoostingRegressor(
    loss = 'ls',
    learning_rate=0.01,
    n_estimators = 1500,
    min_samples_leaf = 140,
    min_samples_split = 300,
    max_depth = 4,
    max_features = 15,
    subsample = 0.9,
    random_state=601
    ),                  
        RandomForestRegressor(
    n_estimators = 200,
    max_depth = 6,
    max_features = 6,
    random_state=601
    )                
    ]

blend_train = np.zeros([X_dev.shape[0],len(clfs)])
blend_test = np.zeros([1000,len(clfs)])
for j, clf in enumerate(clfs):
    print ('Training classifier [%s]' % (j))
    blend_test_j = np.zeros((X_te.shape[0], len(skf))) 
    for i, (train_index, cv_index) in enumerate(skf):
        print ('Fold [%s]' % (i))

        X_train = X_dev[train_index]
        Y_train = Y_dev[train_index]
        X_cv = X_dev[cv_index]
        Y_cv = Y_dev[cv_index]

        clf.fit(X_train, Y_train)
        
        blend_train[cv_index, j] = clf.predict(X_cv)
        blend_test_j[:, i] = clf.predict(X_te)
    blend_test[:, j] = blend_test_j.mean(1)

bclf = LinearRegression()
bclf.fit(blend_train, Y_dev)
print(bclf.coef_)
Y_test_predict = bclf.predict(blend_test)

    
ret = pd.DataFrame()
ret['target'] = np.round(Y_test_predict, 3)
ret.to_csv('../data/test/gbdt_rf_stacking2.csv',index = None,header = None)

    
