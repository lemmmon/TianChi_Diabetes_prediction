# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:46:47 2018
2/26：未加入cv4，线上性能不错
2/27：加入cv4的特征，线上下降
@author: Administrator
"""

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

model1 = xgb.XGBClassifier()

s = 6
l = 200
ans1 = np.zeros([l,s])

#model1
for i in range(s):
    dtr = pd.read_csv('../data/dif_feat/train_data_cv'+str(i+1)+'.csv',encoding = 'gbk')
    lb = pd.read_csv('../data/dif_feat/label_cv'+str(i+1)+'.csv',encoding = 'gbk')  #其实label都是一样的
    dte = pd.read_csv('../data/dif_feat/test_data_cv'+str(i+1)+'.csv',encoding = 'gbk')
    
    #不要调参数，希望每个模型都有一点过拟合，才能检测出baseline检测不到的样本
    
    if i == 1:
        model1 = xgb.XGBClassifier()
    elif i == 2:
        model1 = xgb.XGBClassifier()
    elif i == 3:
        model1 = xgb.XGBClassifier()
    elif i == 4:
        model1 = xgb.XGBClassifier()
    elif i == 5:
       model1 = xgb.XGBClassifier()
    elif i == 6:
        model1 = xgb.XGBClassifier()
    
    model1.fit(dtr,lb)
    ans1[:,i] = model1.predict(dte)       

ans = np.sum(ans1,axis = 1)

for i in range(len(ans)):
    if ans[i]>=3:
        ans[i] = 1
    else:
        ans[i] = 0

ret = pd.DataFrame()
ret['ans'] = ans
ret.to_csv('../data/ans.csv',index = None,header = None)