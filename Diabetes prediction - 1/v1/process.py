#!/usr/bin/env Python
# coding=utf-8
#==============================================================================
# 预处理
#==============================================================================
import pandas as pd
from dateutil.parser import parse
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer

#==============================================================================
#     注意：已事先对男女类别进行编码，每一列名字用数字0-37替换
#==============================================================================
def joint():    #拼接训练集与测试集
    d1 = pd.read_csv('../data/d_train_20180102.csv')
#    d4 = pd.read_csv('D:/TianChi/Tang/data/d_test_A_20180102.csv')  #A榜答案
    d2 = pd.read_csv('../data/d_test_B_20180128.csv')
#    d3 = pd.concat([d1,d4,d2],axis = 0)
    d3 = pd.concat([d1,d2],axis = 0)
    d3 = d3.reset_index(drop=True)
    d3.to_csv('../data/joint.csv',index = None)

def processData():  
    df = pd.read_csv('../data/joint.csv')
    feat_columns = list(df.columns.values)
    feat_columns1 = ['gender','old']
    label = df[['target']]
    label.to_csv('../data/label.csv',index = None)
    df['data'] = (pd.to_datetime(df['data']) - parse('2017-08-09')).dt.days
    for col in feat_columns1:
        df[col] = pd.to_numeric(df[col],errors='coerce')

    feat_columns.remove("id")
    feat_columns.remove("target")
    feat_columns.remove("16")
    feat_columns.remove("17")
    feat_columns.remove("18")
    feat_columns.remove("19")
    feat_columns.remove("20")
    
    df = df.fillna(0,inplace = True)
    data1 = df[feat_columns]

    #增加比值特征
    data1['M/O'] = data1['9'].div(data1['11'])
    data1['J/R'] = data1['6'].div(data1['14'])
    
    #对右倾的数据取对数变换
    data1['1'] = FunctionTransformer(log1p).fit_transform(data1['1'].values)[0] 
    data1['2'] = FunctionTransformer(log1p).fit_transform(data1['2'].values)[0]
    data1['4'] = FunctionTransformer(log1p).fit_transform(data1['4'].values)[0]
    data1['9'] = FunctionTransformer(log1p).fit_transform(data1['9'].values)[0]
    data1['21'] = FunctionTransformer(log1p).fit_transform(data1['21'].values)[0]
    data1['28'] = FunctionTransformer(log1p).fit_transform(data1['28'].values)[0]
    data1['M/O'] = FunctionTransformer(log1p).fit_transform(data1['M/O'].values)[0] 
    data1['J/R'] = FunctionTransformer(log1p).fit_transform(data1['J/R'].values)[0]

    data2 = pd.DataFrame(MinMaxScaler().fit_transform(data1))
    data2.to_csv('../data/data_joint.csv',index = None)   
    
def split_for_submit():    
    data = pd.read_csv('../data/data_joint.csv')
    label = pd.read_csv('../data/label.csv')
    data_tr = data.loc[0:label.shape[0] - 1]
    data_te = data.loc[label.shape[0]:data.shape[0]]
    data_tr = data_tr.reset_index(drop=True)
    data_te = data_te.reset_index(drop=True)
    data_tr.to_csv('../data/data_scale_train.csv',index = None)
    data_te.to_csv('../data/data_scale_test.csv',index = None)  
    
def classLabel(ther):
    label = pd.read_csv('../data/label.csv')
    label[label['target']<ther] = 0
    label[label['target']>ther] = 1
    label.to_csv('../data/label_class_10.csv',index = None)
    
if __name__ == "__main__":
    print("=========================")
    joint()
    processData()
    split_for_submit()
    classLabel(10) #针对不同的值生成label，用于检测异常样本
    print("=========================")
    