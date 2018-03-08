# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:13:57 2018

@author: Administrator
"""

import pandas as pd
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV

dtr = pd.read_csv('../data/dif_feat/train_data_cv6.csv',encoding = 'gbk')
lb = pd.read_csv('../data/dif_feat/label_cv6.csv',encoding = 'gbk')  #其实label都是一样的
dte = pd.read_csv('../data/dif_feat/test_data_cv6.csv',encoding = 'gbk')

#每个子模型，用所有数据进行调参

x_tr = dtr.values
y_tr = lb['label'].iloc[:len(dtr)].values

param_test1 ={} 

gsearch1= GridSearchCV(estimator =XGBClassifier(colsample_bytree = 0.2, subsample = 0.9),   
                      scoring='f1', param_grid =param_test1,cv=10)  
gsearch1.fit(x_tr,y_tr)  

#print(gsearch1.grid_scores_)
print(gsearch1.best_params_) 
print(gsearch1.best_score_)

#model = XGBClassifier()
#
#model.fit(dtr,lb)
#ans = model.predict(dte)
#
#lb_te = pd.read_csv('../data/cv_data/cross_validation_test_1.csv',encoding = 'gbk')
#lb_te = lb_te[['label']].values
#print(f1_score(ans,lb_te))