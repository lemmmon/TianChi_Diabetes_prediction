# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 21:31:16 2018
数据分析
@author: Administrator
"""
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei'] 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats

dt = pd.read_csv('../data/train_data.csv',encoding = 'gbk')

col_list = ['孕前BMI', '糖筛孕周', 'VAR00007', 'wbc',
            'ALT', 'Cr', 'BUN', 'CHO', 'TG', 'HDLC', 'LDLC', 'ApoA1', 'ApoB',
            'hsCRP', '年龄','Lpa']

dt = dt[col_list]

corrmat = dt.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);