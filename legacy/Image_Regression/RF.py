# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:35:01 2018

@author: Luigi Tommaso Luppino
"""
import sklearn.gaussian_process as gp
import numpy as np
import scipy.io
import sys
from sklearn.ensemble import RandomForestRegressor as rf
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trees", default=256, help="number of trees", type=int)
parser.add_argument("--pruning", default=10, help="datapoints per leave", type=int)
args = parser.parse_args()
trees = args.trees
m = args.pruning

mat = scipy.io.loadmat("Data_and_training_sample.mat")
mask = np.array(mat["mask"], dtype=bool)
t1_test = np.array(mat["t1"], dtype=float)
t1_test = np.reshape(t1_test, (-1, t1_test.shape[-1]))
t2_test = np.array(mat["t2"], dtype=float)
t2_test = np.reshape(t2_test, (-1, t2_test.shape[-1]))
idx = np.where(mask)[0]
t1 = t1_test[idx.transpose()]
t2 = t2_test[idx.transpose()]

del mat

regr = rf(
    n_estimators=trees,
    max_features="sqrt",
    oob_score=True,
    n_jobs=-1,
    min_samples_leaf=m,
)

regr.fit(t1, t2)
t1_hat = regr.predict(t1_test)

regr.fit(t2, t1)
t2_hat = regr.predict(t2_test)

d1 = np.linalg.norm(t2_hat - t1_test, axis=1)
outliers = d1 > np.nanmean(d1) + 3 * np.nanstd(d1)
d1[outliers] = np.nanmax(d1[outliers == 0])
d1 = d1 / np.nanmax(d1)
d2 = np.linalg.norm(t1_hat - t2_test, axis=1)
outliers = d2 > np.nanmean(d2) + 3 * np.nanstd(d2)
d2[outliers] = np.nanmax(d2[outliers == 0])
d2 = d2 / np.nanmax(d2)
d = d1 + d2

scipy.io.savemat("RF_output", {"d_RF": d})
