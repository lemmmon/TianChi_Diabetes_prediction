# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:12:23 2018

@author: Administrator
"""
#import model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
#import tools
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

def model_cross_validation(model_name):
    train_data = pd.read_csv('../data/train_data.csv',encoding = 'gbk')
    label = pd.read_csv('../data/label.csv',encoding = 'gbk')
    
    train_data = np.array(train_data.values)
    label = np.array(label.values)
    
    skf = StratifiedKFold(n_splits = 5)
    #初始化模型与参数
    score = 0
    score_sum = 0
    i = 0
    
    if model_name == "xgboost":
        model = xgb.XGBClassifier()
    elif model_name == "GBDT":
        model = GradientBoostingClassifier(random_state = 601)
    elif model_name == "RF":
        model = RandomForestClassifier(random_state = 601)
    elif model_name == "Logistic":
        model = LogisticRegression()
    elif model_name == "SVM":
        model = SVC(kernel='rbf', probability=True) 
    else:
        print("输入参数错误")
    
    #K-fold验证
    for train_index,test_index in skf.split(train_data,label):
        
        dtr, dcv = train_data[train_index], train_data[test_index]
        lb_tr, lb_cv = label[train_index], label[test_index]

        model.fit(dtr,lb_tr)
        
        ans_cv = model.predict(dcv)
        score = metrics.f1_score(lb_cv,ans_cv) #注释
        score_sum = score_sum + score
        print("f1 ["+str(i)+"]"+str(score))
        i = i + 1
        
    print("平均f1: "+str(score_sum/5))
    return score_sum/5
    
    