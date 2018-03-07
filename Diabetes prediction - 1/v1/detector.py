# -*- coding: utf-8 -*-
#==============================================================================
# 用于检测异常血糖样本
#==============================================================================
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from post_process import findIndex

dtr = pd.read_csv('../data/data_scale_train.csv')
lab = pd.read_csv('../data/label_class_10.csv')
predict_set = pd.read_csv('../data/data_scale_test.csv')
dt = pd.concat([dtr,lab],axis = 1)
dt_pos = dt[dt['target'] == 1]
dt_neg = dt[dt['target']== 0]
x_test = dtr.values 
x_predict = predict_set.values
#m = []
for nn in range(50):
    for itr in range(200,701,100):
        clf = GradientBoostingClassifier()
        sub_model_number = itr
        detect_result = np.zeros([len(dtr),sub_model_number]) #
        detect_result2 = np.zeros([len(x_predict),sub_model_number]) #
        
        def splitData(dt):
            y_t = dt['target'].values
            col_name  = list(dt.columns.values)
            col_name.remove("target")
            x_t = dt[col_name].values  
            return x_t, y_t  
        
        for i in range(sub_model_number):
            dt_tr = dt_neg.sample(n=len(dt_pos))    #subsample
            dt = pd.concat([dt_tr,dt_pos],axis = 0).sample(frac=1).reset_index(drop = True)
            x_tr, y_tr = splitData(dt)
            clf.fit(x_tr,y_tr)
            detect_result[:,i] = clf.predict(x_test)
            detect_result2[:,i] = clf.predict(x_predict)
        
        voting_matrix = detect_result.mean(1)
        voting_result = np.zeros([len(lab),1])
        
        voting_matrix2 = detect_result2.mean(1)
        voting_result2 = np.zeros([len(x_predict),1])
        
        def jdg(vot_res):
            for i in range(len(vot_res)):
                if vot_res[i] == 1:
                    vot_res[i] = 1
                else:
                    vot_res[i] = 0
            return vot_res
            
        
        voting_result = jdg(voting_matrix)
        voting_result2 = jdg(voting_matrix2)
        
        ret = pd.DataFrame()
        ret['label'] = voting_result2
        ret.to_csv('../data/test/high_value'+str(itr)+'.csv',index = None)
    
    a = findIndex(5)
    print(a)    #打印异常样本的标签，用于改值
        
    
    
    
    