# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:08:35 2018

@author: Administrator
"""
import pandas as pd

#四则运算连续特征
def generateNewFeature_1(dt,col,col_continue):
    l = len(col)
    for i in range(l):
        for j in range(i,l):
            c1 = col[i]
            c2 = col[j]
            if c1 not in col_continue or c2 not in col_continue:
                dt[c1+"*"+c2] = dt[c1].mul(dt[c2])
                col.append(c1+"*"+c2)
            
            
    print("已生成四则运算特征")
    dt = dt.replace(float('inf'),0)  #替换无穷大
    dt.fillna(0,inplace = True)
    return dt


#将特征加入原始数据
def addNewFeature(dt,cv_flag):
    if cv_flag == 1:    #cv_data
        return dt
    elif cv_flag == 2:
        return dt
    elif cv_flag == 3:
        dt['AST*SNP34_2'] = dt['AST'].mul(dt['SNP34_2'])
        dt['SNP26_2*SNP46_1'] = dt['SNP26_2'].mul(dt['SNP46_1'])
        dt['SNP3_0*SNP21_0'] = dt['SNP3_0'].mul(dt['SNP21_0'])
        dt['hsCRP*SNP37_3'] = dt['hsCRP'].mul(dt['SNP37_3'])
        dt['wbc*SNP5_0'] = dt['wbc'].mul(dt['SNP5_0'])
        dt['年龄*SNP46_0'] = dt['年龄'].mul(dt['SNP46_0'])
        dt['糖筛孕周*SNP46_2'] = dt['糖筛孕周'].mul(dt['SNP46_2'])
        dt['SNP34_2*SNP37_2'] = dt['SNP34_2'].mul(dt['SNP37_2'])
        dt['ApoB*SNP37_2'] = dt['ApoB'].mul(dt['SNP37_2'])
        dt['Cr*SNP39_2'] = dt['Cr'].mul(dt['SNP39_2'])
        dt['LDLC*SNP5_0'] = dt['LDLC'].mul(dt['SNP5_0'])
        dt['DM家族史_0*SNP48_1'] = dt['DM家族史_0'].mul(dt['SNP48_1'])
        dt['wbc*SNP15_0'] = dt['wbc'].mul(dt['SNP15_0'])
        dt['CHO*SNP29_2'] = dt['CHO'].mul(dt['SNP29_2'])
         
    elif cv_flag == 4:
        dt['身高*SNP20_0'] = dt['身高'].mul(dt['SNP20_0'])
        dt['AST*SNP34_2'] = dt['AST'].mul(dt['SNP34_2'])
        dt['孕前BMI*SNP53_0'] = dt['孕前BMI'].mul(dt['SNP53_0'])
        dt['SNP34_2*SNP55_3'] = dt['SNP34_2'].mul(dt['SNP55_3'])
        dt['hsCRP*SNP24_0'] = dt['hsCRP'].mul(dt['SNP24_0'])
        dt['身高*SNP31_0'] = dt['身高'].mul(dt['SNP31_0'])
        dt['wbc*SNP5_0'] = dt['wbc'].mul(dt['SNP5_0'])
        dt['糖筛孕周*SNP46_2'] = dt['糖筛孕周'].mul(dt['SNP46_2'])
        dt['hsCRP*SNP37_3'] = dt['hsCRP'].mul(dt['SNP37_3'])
        dt['SNP31_1*SNP37_2'] = dt['SNP31_1'].mul(dt['SNP37_2'])
        dt['BUN*SNP17_3'] = dt['BUN'].mul(dt['SNP17_3'])
        dt['年龄*SNP46_0'] = dt['年龄'].mul(dt['SNP46_0'])
        dt['Lpa*SNP34_2'] = dt['Lpa'].mul(dt['SNP34_2'])
        dt['SNP23_1*SNP34_1'] = dt['SNP23_1'].mul(dt['SNP34_1'])
   
    elif cv_flag == 5:
        dt['VAR00007*SNP46_0'] = dt['VAR00007'].mul(dt['SNP46_0'])
        dt['SNP34_2*SNP52_1'] = dt['SNP34_2'].mul(dt['SNP52_1'])
        dt['SNP15_1*SNP23_2'] = dt['SNP15_1'].mul(dt['SNP23_2'])
        dt['收缩压*SNP38_1'] = dt['收缩压'].mul(dt['SNP38_1'])
        dt['年龄*SNP11_1'] = dt['年龄'].mul(dt['SNP11_1'])
        dt['孕前BMI*SNP53_0'] = dt['孕前BMI'].mul(dt['SNP53_0'])
        dt['SNP31_3*SNP39_1'] = dt['SNP31_3'].mul(dt['SNP39_1'])
        dt['Lpa*SNP20_0'] = dt['Lpa'].mul(dt['SNP20_0'])
        dt['TG*SNP53_1'] = dt['TG'].mul(dt['SNP53_1'])

    elif cv_flag == 6:
        dt['ApoB*SNP20_0'] = dt['ApoB'].mul(dt['SNP20_0'])
        dt['SNP21_0*SNP3_0'] = dt['SNP21_0'].mul(dt['SNP3_0'])
        dt['SNP34_2*SNP55_3'] = dt['SNP34_2'].mul(dt['SNP55_3'])
        dt['年龄*SNP46_0'] = dt['年龄'].mul(dt['SNP46_0'])
        dt['SNP15_1*SNP23_2'] = dt['SNP15_1'].mul(dt['SNP23_2'])
        dt['SNP28_2*SNP40_1'] = dt['SNP28_2'].mul(dt['SNP40_1'])
        dt['SNP34_2*SNP37_2'] = dt['SNP34_2'].mul(dt['SNP37_2'])
        dt['hsCRP*SNP37_3'] = dt['hsCRP'].mul(dt['SNP37_3'])
        dt['孕前BMI*SNP53_0'] = dt['孕前BMI'].mul(dt['SNP53_0'])
        dt['SNP21_1*SNP34_2'] = dt['SNP21_1'].mul(dt['SNP34_2'])
        dt['SNP43_1*SNP49_2'] = dt['SNP43_1'].mul(dt['SNP49_2'])
        dt['hsCRP*SNP6_1'] = dt['hsCRP'].mul(dt['SNP6_1'])
        dt['wbc*SNP5_0'] = dt['wbc'].mul(dt['SNP5_0'])
        dt['AST*SNP34_2'] = dt['AST'].mul(dt['SNP34_2'])

#
    dt = dt.replace(float('inf'),0)  #替换无穷大
    dt.fillna(0,inplace = True)
    return dt

def one_hot(dt,col_continue,col_dist):
    dt_dist = dt[col_dist]
    dt_cod = pd.DataFrame()
    for c in col_dist:
        c_d = pd.get_dummies(dt_dist[c])
        col_name = []
        for i in range(c_d.shape[1]):
            col_name.append(c+"_"+str(i))
        c_d.columns = col_name
        dt_cod = pd.concat([dt_cod,c_d],axis = 1)
    
    dt_cod = pd.concat([dt[col_continue],dt_cod],axis = 1)
    return dt_cod





