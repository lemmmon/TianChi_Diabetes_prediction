# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
#load data
reader = pd.read_csv('D:/TianChi/Tang/data/data_scale_train.csv',chunksize = 10000000) 
model = []       
for chunk in reader:
    df = pd.DataFrame(chunk)
    model.append(df)
X_train = pd.concat(model)

reader = pd.read_csv('D:/TianChi/Tang/data/label.csv',chunksize = 10000000) 
model = []       
for chunk in reader:
    df = pd.DataFrame(chunk)
    model.append(df)
label = pd.concat(model)


m = []
#for i in range(1):
#x_tr, x_te, y_tr, y_te = train_test_split(X_train, label['target'], test_size=0.2)
x_tr = X_train
y_tr = label['target']
#
# GBDT模型
##搜索参数
param_test1 ={'n_estimators':range(100,201,20)} 
gsearch1= GridSearchCV(estimator =RandomForestRegressor(
                                                            loss = 'ls',
                                                            learning_rate=0.03,
                                                            n_estimators = 140,
                                                            min_samples_split=140,
                                                            min_samples_leaf=300,
                                                            max_depth=4,
                                                            max_features=12, 
                                                            subsample=1,
                                                            random_state=610

                                                            ),   
                      scoring='neg_mean_squared_error', param_grid =param_test1,cv=5)  
gsearch1.fit(x_tr,y_tr)  
print(gsearch1.grid_scores_)
print(gsearch1.best_params_) 
print(gsearch1.best_score_)

#####################不要改动参数了###########################
#gbdt = GradientBoostingRegressor(   #第三版参数，更多的特征
#            loss = 'ls',
#            learning_rate=0.01,
#            n_estimators = 1000,
#            min_samples_split=140,
#            min_samples_leaf=300,
#            max_depth=4,
#            max_features=15, 
#            subsample=0.85,
#            random_state=610
#        )   
#    gbdt = GradientBoostingRegressor(#xinban
#        loss = 'ls',
#        learning_rate=0.01,        #第二版参数 0.05，300
#        n_estimators = 1500, 
#        min_samples_split=10,
#        min_samples_leaf=110,
#        max_depth=11,
#        max_features='sqrt', 
#        subsample=0.8,
#        random_state=610
#        )
#gbdt = RandomForestRegressor(
#                            n_estimators = 500,
#                            min_samples_split=140,
#                            min_samples_leaf=300,
#                            max_features = 'auto',
#                            random_state=610
#                            )
#gbdt.fit(x_tr,y_tr)

####################################################
#reader = pd.read_csv('D:/TianChi/Tang/data/data_scale_test.csv',chunksize = 10000000) 
#model = []       
#for chunk in reader:
#    df = pd.DataFrame(chunk)
#model.append(df)
#submit = pd.concat(model)
#
#
#label_submit = gbdt.predict(submit)
##label_sub = np.round(label_submit,3)    #保留三位小数
#ret = pd.DataFrame()
#
#ret['target'] = label_submit
#ret.to_csv('D:/TianChi/Tang/data/test/rf_final.csv', index=False,header = True)

######################################################
#def loss(list1, list2):
#    import math
#    _sum = 0
#    for k, v in zip(list1, list2):
#        _sum += math.pow(k-v, 2)
#    return _sum / len(list1)/2
#
#label_test = gbdt.predict(x_te)
#m.append(loss(label_test,y_te))
#print(loss(label_test,y_te))
#print('average loss:' + str(np.mean(m)))
