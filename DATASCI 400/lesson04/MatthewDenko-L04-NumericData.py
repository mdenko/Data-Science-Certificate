#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:07:48 2018

@author: matt.denko
"""

"Import statements"
"Load your dataset"
"Assign reasonable column names, the data set description"
"Median imputation of the missing numeric values"
"Outlier replacement if applicable"
"Histogram of a numeric variable. Use plt.show() after each histogram"
"Create a scatterplot. Use plt.show() after the scatterplot"
"Determine the standard deviation of all numeric variables. Use print() for each standard deviation"
"Comments explaining the code blocks"
"Summary comment block on how the numeric variables have been treated: which ones had outliers, required imputation, distribution, removal of rows/columns."
#------------------------------------------------------------------------------
"import statements"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
"loading the data set from my documents folder"

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

Adult= pd.read_csv(url, header=None)

print(Adult)
#------------------------------------------------------------------------------
"Assigning reasonable column names"

Adult.columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
                 "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country",">50K, <=50k"]

print(Adult)
#------------------------------------------------------------------------------
"Median imputation of the missing numeric columns"

#checking data types

adult_data_types = Adult.dtypes

print(adult_data_types)
#replacing ? wiht Nan for numeric columns

Adult = Adult.replace(to_replace= "?", value=float("NaN"))

print(Adult)

#suming Nans

Adult_null = Adult.isnull().sum()

print(Adult_null)

###There are 0 Nans for numeric columns, Below is the code I would execute if there were
#Adult.loc[HasNan, "age"] =  np.nanmedian(Heart.loc[:,"agel"])

#------------------------------------------------------------------------------
"Outlier replacement if applicable"

###checking distribution for outliers 

#Age

age_hist = plt.hist(Adult.loc[:,'age'])

plt.show(age_hist)

#fnlwgt

fnlwgt_hist = plt.hist(Adult.loc[:,'fnlwgt'])

plt.show(fnlwgt_hist)

#education-num

education_num_hist = plt.hist(Adult.loc[:,'education-num'])

plt.show(education_num_hist)

#capital-gain

capital_gain_hist = plt.hist(Adult.loc[:,'capital-gain'])

plt.show(capital_gain_hist)

#capital-loss

capital_loss_hist = plt.hist(Adult.loc[:,'capital-loss'])

plt.show(capital_loss_hist)

#hours-per-week

hours_per_week_hist = plt.hist(Adult.loc[:,'hours-per-week'])

plt.show(hours_per_week_hist)

###based off the distribution of age I will replace all values that are >55 with the median

# Replace outlier with median

TooHigh = Adult.loc[:, "age"] > 55

print(TooHigh)

Adult.loc[TooHigh, "age"] = np.median(Adult.loc[:,"age"])

print(Adult)

#------------------------------------------------------------------------------
"Histogram of a numeric variable. Use plt.show() after each histogram"

#rechecking the distribution of age after removing outliers

age_hist = plt.hist(Adult.loc[:,'age'])

plt.show(age_hist)

#------------------------------------------------------------------------------
"Create a scatterplot. Use plt.show() after the scatterplot"

#creating the scatter plot to see relationship between age and capital gain

scatter_plot = plt.scatter(Adult.loc[:,'age'],Adult.loc[:,"capital-gain"]) 

plt.show(scatter_plot)

#------------------------------------------------------------------------------
"Determine the standard deviation of all numeric variables. Use print() for each standard deviation"

#Age

age_std = np.std(Adult.loc[:,"age"])

print(age_std)

#fnlwgt

fnlwgt_std = np.std(Adult.loc[:,"fnlwgt"])

print(fnlwgt_std)

#education-num

education_num_std = np.std(Adult.loc[:,"education-num"])

print(education_num_std)

#capital-gain

capital_gain_std = np.std(Adult.loc[:,"capital-gain"])

print(capital_gain_std)

#capital-loss

capital_loss_std = np.std(Adult.loc[:,"capital-loss"])

print(capital_loss_std)

#hours-per-week

hours_per_week_std = np.std(Adult.loc[:,"hours-per-week"])

print(hours_per_week_std)

#------------------------------------------------------------------------------
"Summary comment block on how the numeric variables have been treated: which ones had outliers, required imputation, distribution, removal of rows/columns."


"""There are 6 numeric variables in this data set. None of the numeric columns
contained missing or null values. I plotted the distribution of each numeric 
variable and noticed that the age variable was skewed right so I applied a 
judgement call based off the distribution to remove any age >55. After removing
these the distribution was much closer to normal. Outside of that I did not 
remove any outliers""" 

