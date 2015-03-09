# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:04:38 2015

@author: tdoughty1
"""

from pandas import read_csv
from sklearn.linear_model import LogisticRegression

train = read_csv('data/train.csv', parse_dates=[0])
test = read_csv('data/test.csv', parse_dates=[0])

trainArray = train.as_matrix()

X = trainArray[:, 1:9]
y = trainArray[:,-1]

fitter = LogisticRegression()
fitter.fit(X,y)
