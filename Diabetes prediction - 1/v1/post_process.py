# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#==============================================================================
# get intersection from severeal predict
#==============================================================================
def findIndex(threshold):

    d2 = pd.read_csv('../data/test/high_value200.csv')
    d3 = pd.read_csv('../data/test/high_value300.csv')
    d4 = pd.read_csv('../data/test/high_value400.csv')
    d5 = pd.read_csv('../data/test/high_value500.csv')
    d6 = pd.read_csv('../data/test/high_value600.csv')
    d7 = pd.read_csv('../data/test/high_value700.csv')

    dv2 = d2['label'].values
    dv3 = d3['label'].values
    dv4 = d4['label'].values
    dv5 = d5['label'].values
    dv6 = d6['label'].values
    dv7 = d7['label'].values

    r = np.zeros([len(dv2),1])
    ind = []
    for i in range(len(dv2)):
        count = 0
        if dv2[i] == 1:
            count = count + 1
        if dv3[i] == 1:
            count = count + 1
        if dv4[i] == 1:
            count = count + 1
        if dv5[i] == 1:
            count = count + 1
        if dv6[i] == 1:
            count = count + 1
        if dv7[i] == 1:
            count = count + 1
            
        if count>=threshold:
            r[i] = 1
            ind.append(i + 1)    #
    return ind
        