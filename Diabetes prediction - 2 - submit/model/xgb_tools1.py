# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei'] 

import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

def xgb_single_model(method):
    train_data = pd.read_csv('../data/train_data.csv',encoding = 'gbk')
    test_data = pd.read_csv('../data/test_data.csv',encoding = 'gbk')
    label = pd.read_csv('../data/label.csv',encoding = 'gbk')
    
    if method == 'bagging': #性能一般，可能是因为数据集太小了
        dtr   = np.array(train_data.values)
        lb_tr = np.array(label.iloc[0:len(dtr)].values)
        lb_te = np.array(label.iloc[len(dtr):].values)
        predict_data = np.array(test_data.values)
        
        model = xgb.XGBClassifier()
        splits_number = 50
        skf = StratifiedKFold(n_splits = splits_number)
        ans = np.zeros([len(lb_te),splits_number])
        final_ans = np.zeros([len(lb_te),1])
        i = 0
        
        for train_index,test_index in skf.split(dtr,lb_tr):
        
            data_train, data_cv = dtr[train_index], dtr[test_index]
            label_train, label_cv = lb_tr[train_index], lb_tr[test_index]
        
            model.fit(data_train,label_train)
            ans[:,i] = model.predict(predict_data)
            i = i + 1
            
        ans_mean = ans.mean(1)
        
        for i in range(len(ans_mean)):
            if ans_mean[i] > 0.5:
                final_ans[i,0] = 1
            else:
                final_ans[i,0] = 0
                
        print('score:'+str(f1_score(final_ans,lb_te)))
        return f1_score(final_ans,lb_te)
    
    if method == 'single':
        dtr   = train_data
        lb_tr = label.iloc[0:len(dtr)]
        lb_te = label.iloc[len(dtr):]
        
        model = xgb.XGBClassifier()
        model.fit(dtr,lb_tr)
        ans = model.predict(test_data)
        lb_te = np.array(lb_te.values)
#        for i in range(len(ans)):
#            if ans[i] != lb_te[i]:
#                print(str(ans[i])+' '+str(lb_te[i]))
        
        print('score:'+str(f1_score(lb_te,ans)))
        
        return f1_score(lb_te,ans)








