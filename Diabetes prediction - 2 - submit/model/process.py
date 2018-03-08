# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:50:38 2018

@author: Hao Ying
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import new_Feature as nF
import model_cross_validation as mcv
import numpy as np
from xgb_tools1 import xgb_single_model

def concatDt(d1,d2):    
    dt = pd.concat([d1,d2],axis = 0)
    dt = dt.reset_index(drop = True)
    return dt

def splitDt(dt,train_row,test_row):
    d1 = dt.iloc[0:train_row]
    d2 = dt.iloc[train_row:]
    d1 = d1.reset_index(drop = True)
    d2 = d2.reset_index(drop = True)
    return d1,d2  

def fillNa(d1,d2,col):
    col_continue = ["孕前BMI","收缩压","舒张压","分娩时","糖筛孕周","VAR00007","wbc",
                "ALT","AST","Cr","BUN","CHO","TG","HDLC","LDLC","ApoA1","ApoB",
                "hsCRP","身高","孕前体重","RBP4","年龄","Lpa"]
    for c in col_continue:
        col.remove(c)
    col_dist = col
    d1_continue = d1[col_continue]
    d1_dist = d1[col_dist]
    d2_continue = d2[col_continue]
    d2_dist = d2[col_dist]

    d1_continue.fillna(d1_continue.mean(),inplace = True)#连续特征不可填0
    d1_dist.fillna(0,inplace = True)
    d2_continue.fillna(d1_continue.mean(),inplace = True)#连续特征不可填0
    d2_dist.fillna(0,inplace = True)
        
    mns = MinMaxScaler()
    d1_continue_scale = pd.DataFrame(mns.fit_transform(d1_continue),columns = col_continue)
    d2_continue_scale = pd.DataFrame(mns.transform(d2_continue),columns = col_continue)

#    
    d1 = pd.concat([d1_continue_scale,d1_dist],axis = 1)
    d2 = pd.concat([d2_continue_scale,d2_dist],axis = 1)
    return d1,d2,col_continue,col_dist

def pre_process(d1,d2,cv_flag):

    dt = concatDt(d1,d2)

    col_list = list(dt.columns.values)
#    dt[['label']].to_csv('../data/label.csv',index = None)
    dt[['label']].to_csv('../data/dif_feat/label_cv'+str(cv_flag)+'.csv',index = None)
#    
    col_list.remove("label")
    col_list.remove("id")
    dt = dt[col_list]
    
    #移除离散特征
    col_list.remove('产次')#论文说无效
    
    col_list.remove('SNP1')#p值较高
    col_list.remove('SNP4')
    col_list.remove('SNP14')
    col_list.remove('SNP44')
    col_list.remove('ACEID')
    
    dt = dt[col_list]
    
    #填充缺失值
    d1,d2 = splitDt(dt,len(d1),len(d2))
    d1,d2,col_continue,col_dist = fillNa(d1,d2,col_list)    #类别填0连续变量填均值，啥都不填是最好的
    dt = concatDt(d1,d2)

    #对类别进行编码
    dt = nF.one_hot(dt,col_continue,col_dist)
    ##########组合新的特征#######
    dt = nF.addNewFeature(dt,cv_flag)
    col_list = list(dt.columns.values)  #重新获得header
    ##删除连续特征       
    #缺失较大的连续型数据
    col_list.remove('RBP4')
    col_list.remove('分娩时')
    col_list.remove('AST')
    col_list.remove('舒张压')
    col_list.remove('收缩压')#舒张压与收缩压共线性
    col_list.remove('孕前体重') #与BMI有关，删掉两个特征
    col_list.remove('身高')   #与身高无关
    
    dt = dt[col_list]
    
    print(dt.shape[1])
    
    dtr,dts = splitDt(dt,len(d1),len(d2))
    
#    dtr.to_csv('../data/train_data.csv',index = None,encoding = 'gbk')
#    dts.to_csv('../data/test_data.csv', index = None,encoding = 'gbk')
    
    dtr.to_csv('../data/dif_feat/train_data_cv'+str(cv_flag)+'.csv',index = None,encoding = 'gbk')
    dts.to_csv('../data/dif_feat/test_data_cv'+str(cv_flag)+'.csv', index = None,encoding = 'gbk')
#    
if __name__ == "__main__":
##    测试集得分
    print("===========开始============")
    d1 = pd.read_csv('../data/f_train_20180204.csv',encoding = 'gbk')
    
    d1_t = pd.read_csv('../data/f_test_a_20180204.csv',encoding = 'gbk')
    d1_t_ans = pd.read_csv('../data/f_answer_a_20180306.csv',encoding = 'gbk',header = None)
    d1_t['label'] = d1_t_ans.values
    d1 = pd.concat([d1,d1_t],axis = 0).reset_index(drop = True)
    print(len(d1))
    d2 = pd.read_csv('../data/f_test_b_20180305.csv',encoding = 'gbk')

    print("=========已读入数据=========")
    for cv_flag in range(6):
        pre_process(d1,d2,cv_flag+1)     
    print("========已生成特征==========")
    
    print("============结束============")

#    
    