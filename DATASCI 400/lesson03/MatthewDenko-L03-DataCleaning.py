#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 20:13:29 2018

@author: matt.denko
"""
####Import statements for necessary package(s)--------------------------------

import numpy as np

###Create 3 different numpy arrays with at least 30 items each-----------------

x = np.array([1,2,3,4,5,6,7,8,500,10,11,12,13,-400,15,16,17,18,19,20,21,22,23,24,25,26,27,100,28,50,29,900,2,3,5])

y = np.array([8,9,43,4,15,26,70,8,5,10,11,12,13,-4,15,16,117,18,9,2,21,22,3,24,25,6,27,10,28,50,29,91,2,3,-7])

z = np.array([12,24,"7",54,54,6,75,8,509,10," ",1,13,-400," ",16,1,1,-19,20,21,2,123,24,12,2,7,100,28,50,29,9,20,3,10])

###Write function(s) that remove outliers in the first array-------------------

x = x[(x < np.mean(x) + 2*np.std(x)) & (x > np.mean(x) - 2*np.std(x))]

####Write function(s) that replace outliers in the second array----------------

# calculate the limits for values that are not outliers
LimitHi = np.mean(y) + 2*np.std(y)
LimitLo = np.mean(y) - 2*np.std(y)

# Create Flag for values outside of limits
FlagBad = (y < LimitLo) | (y > LimitHi)

# Replace outliers with mean
y[FlagBad] = np.mean(y)

###Write function(s) that fill in missing values in the third array------------

# Create Flag for missing values

zFlagGood = (z != "?") & (z != " ")

# Create Replacement Value and replace missing values

ReplacementValue = 0

z[~zFlagGood] = ReplacementValue


###Summary comment block on how your dataset has been cleaned u -------

###I cleaned dataset x by removing all values that are plus or minus two standard deviations from the mean.
###These are values that might skew a dataset. I cleaned dataset y by replacing all outliers with the mean value of
###the array. This is useful if you want to remove skewed values but not affect the number of observations. I cleaned
####type if needed.
