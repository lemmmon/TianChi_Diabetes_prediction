# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
import winsound
import pandas as pd
import xgboost as xgb

from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

train_data = pd.read_csv('../data/train_data.csv',encoding = 'gbk')
test_data = pd.read_csv('../data/test_data.csv',encoding = 'gbk')
label = pd.read_csv('../data/label.csv',encoding = 'gbk')

x_tr   = train_data
y_tr = label.iloc[:len(train_data)]

model = xgb.XGBClassifier()
model.fit(x_tr,y_tr)

feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
ret = pd.DataFrame(feat_imp,columns = ['importance'])
ret.to_csv('../data/feature_importance/cv_3.csv',encoding = 'gbk')
winsound.Beep(600,1000)

