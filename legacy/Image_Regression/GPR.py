# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:35:01 2018

@author: Luigi Tommaso Luppino
"""
import sklearn.gaussian_process as gp
import sklearn.preprocessing as pre
import numpy as np
import scipy.io
import cPickle
import sys
import time

# RationalQuadratic = gp.kernels.RationalQuadratic()
regr1 = gp.GaussianProcessRegressor(alpha=1e-5, n_restarts_optimizer=5)
regr2 = gp.GaussianProcessRegressor(alpha=1e-5, n_restarts_optimizer=5)

# Read data
mat = scipy.io.loadmat("Data_and_training_sample.mat")
mask = np.array(mat["mask"], dtype=bool)
t1_test = np.array(mat["t1"], dtype=float)
t1_test = np.reshape(t1_test, (-1, t1_test.shape[-1]))
t2_test = np.array(mat["t2"], dtype=float)
t2_test = np.reshape(t2_test, (-1, t2_test.shape[-1]))
idx = np.where(mask)[0]

# Normalise data
t1 = pre.robust_scale(t1)
t2 = pre.robust_scale(t2)
t1_tr = t1[idx.transpose()]
t2_tr = t2[idx.transpose()]
del mat

regr1.fit(t1_tr, t2_tr)
t1_hat = np.empty((0, t2.shape[1]))
tic = time.time()
for i in range(1 + t1.shape[0] / 31000):
    a = i * 31000
    b = a + 31000
    temp1 = t1[a:b, :]
    t1_hat = np.append(t1_hat, regr1.predict(temp1), axis=0)
    toc = time.time()
    et = toc - tic
    remaining = (t1.shape[0] / 31000 - i) * et / (i + 1)
    print "************"
    print "Elapsed time: ", et // 3600, " hours, ", (et % 3600) // 60, " minutes, ", (
        et % 3600
    ) % 60, " seconds"
    print "Average time. ", ((et / (i + 1)) % 3600) // 60, " minutes, ", (
        (et / (i + 1)) % 3600
    ) % 60, " seconds"
    print "Time to go: ", remaining // 3600, " hours, ", (
        remaining % 3600
    ) // 60, " minutes, ", (remaining % 3600) % 60, " seconds"
    print "************"
del regr1, temp1

regr2.fit(t2_tr, t1_tr)
t2_hat = np.empty((0, t1.shape[1]))
tic = time.time()
for i in range(1 + t2.shape[0] / 31000):
    a = i * 31000
    b = a + 31000
    temp2 = t2[a:b, :]
    t2_hat = np.append(t2_hat, regr2.predict(temp2), axis=0)
    toc = time.time()
    et = toc - tic
    remaining = (t2.shape[0] / 31000 - i) * et / (i + 1)
    print "************"
    print "Elapsed time: ", et // 3600, " hours, ", (et % 3600) // 60, " minutes, ", (
        et % 3600
    ) % 60, " seconds"
    print "Average time. ", ((et / (i + 1)) % 3600) // 60, " minutes, ", (
        (et / (i + 1)) % 3600
    ) % 60, " seconds"
    print "Time to go: ", remaining // 3600, " hours, ", (
        remaining % 3600
    ) // 60, " minutes, ", (remaining % 3600) % 60, " seconds"
    print "************"

del regr2, temp2

d1 = np.linalg.norm(t1_hat - t2, axis=1)
outliers = d1 > np.nanmean(d1) + 3 * np.nanstd(d1)
d1[outliers] = np.nanmax(d1[outliers == 0])
d1 = d1 / np.nanmax(d1)
d2 = np.linalg.norm(t2_hat - t1, axis=1)
outliers = d2 > np.nanmean(d2) + 3 * np.nanstd(d2)
d2[outliers] = np.nanmax(d2[outliers == 0])
d2 = d2 / np.nanmax(d2)
d = d1 + d2
scipy.io.savemat("GPR_output", {"d_GPR": d})
