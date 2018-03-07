# -*- coding: utf-8 -*-

import pandas as pd
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

for i in range(10):
    x_tr, x_te, y_tr, y_te = train_test_split(X_train, label['target'], test_size=0.2)
#    x_tr = X_train
#    y_tr = label['target']
    
    # RF模型
    #搜索参数
    #param_test1 ={'max_depth':list(range(2,13,2))} 
    #gsearch1= GridSearchCV(estimator =RandomForestRegressor(
    #                                                        n_estimators = 60,
    #                                                        max_depth = 8,
    #                                                        max_features = 15,
    #                                                        min_samples_leaf = 140,
    #                                                        min_samples_split = 300,
    #                                                        random_state=601),   
    #                      scoring='neg_mean_squared_error', param_grid =param_test1,cv=5)  
    #gsearch1.fit(x_tr,y_tr)  
    #print(gsearch1.grid_scores_)
    #print(gsearch1.best_params_) 
    #print(gsearch1.best_score_)
    ####################################################
    rf=RandomForestRegressor(
        n_estimators = 600,
        max_depth = 8,
        max_features = 15,
        min_samples_leaf = 140,
        min_samples_split = 300,
        random_state=601
        )
    rf.fit(x_tr,y_tr)
    
    ######################################################
    #reader = pd.read_csv('D:/TianChi/Tang/data/data_scale_test.csv',chunksize = 10000000) 
    #model = []       
    #for chunk in reader:
    #    df = pd.DataFrame(chunk)
    #    model.append(df)
    #submit = pd.concat(model)
    #
    #
    #label_submit = rf.predict(submit)
    ##label_sub = np.round(label_submit,3)    #保留三位小数
    #ret = pd.DataFrame()
    #ret['target'] = label_submit
    #ret.to_csv('D:/TianChi/Tang/data/syx/result_RF_1_13.csv', index=False)
    
    ######################################################
    def loss(list1, list2):
        import math
        _sum = 0
        for k, v in zip(list1, list2):
            _sum += math.pow(k-v, 2)
        return _sum / len(list1)/2
    
    label_test = rf.predict(x_te)
    print(loss(label_test,y_te))