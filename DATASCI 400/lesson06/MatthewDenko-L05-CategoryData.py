#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 08:15:49 2018

@author: matt.denko
"""

"""Import statements
Loading your dataset
Normalize one variable
Bin one variable
Decoding categorical data
Imputing missing values
Consolidating categories if applicable
One-hot encoding (dummy variables) for a categorical column with more than 2 categories
New columns created, obsolete deleted if applicable
Plots for 1 or more categories
Comments explaining the code blocks
Summary comment block on how the categorical data has been treated: decoded, imputed, consolidated, dummy variables created."""

#-------------------------------------------------------------------------------
"import statements"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
"loading the data set"

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

Adult= pd.read_csv(url, header=None)

print(Adult)

#Assigning Column Names

Adult.columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
                 "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country",">50K, <=50k"]

print(Adult)

#replacing ? wiht Nan for numeric columns

Adult = Adult.replace(to_replace= "?", value=float("NaN"))

print(Adult)

#suming Nans

Adult_null = Adult.isnull().sum()

print(Adult_null)

###There are 0 Nans for numeric columns

#------------------------------------------------------------------------------
"Normalize one variable"

# Extracting the Age Column

Age = Adult.loc[:,'age']

# Normalizing the age variable using numpy and z normalization

Age_zscaled = (Age - np.mean(Age))/np.std(Age)

#Comparing the distribution of Age before and after normalization

age_hist = plt.hist(Age)

plt.show(age_hist)

age_zscaled_hist = plt.hist(Age_zscaled)

plt.show(age_hist)

### the distribution did not change, only the spread

#------------------------------------------------------------------------------
"Bin one variable"

### I want to bin hours per week into three bins of equivalent size L, M, H

hpw = Adult.loc[:,"hours-per-week"]

# Equal-width Binning using numpy
NumberOfBins = 3
BinWidth = (max(hpw) - min(hpw))/NumberOfBins
MinBin1 = float('-inf')
MaxBin1 = min(hpw) + 1 * BinWidth
MaxBin2 = min(hpw) + 2 * BinWidth
MaxBin3 = float('inf')

print("\n########\n\n Bin 1 is from ", MinBin1, " to ", MaxBin1)
print(" Bin 2 is greater than ", MaxBin1, " up to ", MaxBin2)
print(" Bin 3 is greater than ", MaxBin2, " up to ", MaxBin3)

Binned_EqW = np.array([" "]*len(hpw)) # Empty starting point for equal-width-binned array
Binned_EqW[(MinBin1 < hpw) & (hpw <= MaxBin1)] = "L" # Low
Binned_EqW[(MaxBin1 < hpw) & (hpw <= MaxBin2)] = "M" # Med
Binned_EqW[(MaxBin2 < hpw) & (hpw  < MaxBin3)] = "H" # High

print(" x binned into 3 equal-width bins: ")
print(Binned_EqW)

#------------------------------------------------------------------------------
"Decoding categorical data"

# Check the data types

data_types = Adult.dtypes

print(data_types)

# Check the first rows of the data frame

Adult.head()

# Check the unique values for all numeric types

## age

age_unique = Adult.loc[:, "age"].unique()

print(age_unique)

## fnlwgt

fnlwgt_unique = Adult.loc[:, "fnlwgt"].unique()

print(fnlwgt_unique)

## education-num

education_unique = Adult.loc[:, "education-num"].unique()

print(education_unique)

## capital-gain

capitalgain_unique = Adult.loc[:, "capital-gain"].unique()

print(capitalgain_unique)

## capital-loss

capitalloss_unique = Adult.loc[:, "capital-loss"].unique()

print(capitalloss_unique)

## hours-per-week

hours_unique = Adult.loc[:, "hours-per-week"].unique()

print(hours_unique)

####Based off these results the only value that makes sense to decode is
##education-num, I am aware that there is already an education column for
#education-num but I will still decode education-num for the assigment

### Decode education-num

## convert to object
Education_num = Adult.loc[:, "education-num"]

Adult.loc[:, "education-num"] = Education_num.astype(object)

## Replace Values

Replace1 = Adult.loc[:,"education-num"] == 1
Adult.loc[Replace1, "education-num"] ="Elementary"

Replace2 = Adult.loc[:,"education-num"] == 2
Adult.loc[Replace2, "education-num"] ="Elementary"

Replace3 = Adult.loc[:,"education-num"] == 3
Adult.loc[Replace3, "education-num"] ="Elementary"

Replace4 = Adult.loc[:,"education-num"] == 4
Adult.loc[Replace4, "education-num"] ="Elementary"

Replace5 = Adult.loc[:,"education-num"] == 5
Adult.loc[Replace5, "education-num"] ="Elementary"

Replace6 = Adult.loc[:,"education-num"] == 6
Adult.loc[Replace6, "education-num"] ="Elementary"

Replace7 = Adult.loc[:,"education-num"] == 7
Adult.loc[Replace7, "education-num"] ="Primary"

Replace8 = Adult.loc[:,"education-num"] == 8
Adult.loc[Replace8, "education-num"] ="Primary"

Replace9 = Adult.loc[:,"education-num"] == 9
Adult.loc[Replace9, "education-num"] ="Primary"

Replace10 = Adult.loc[:,"education-num"] == 10
Adult.loc[Replace10, "education-num"] ="Primary"

Replace11 = Adult.loc[:,"education-num"] == 11
Adult.loc[Replace11, "education-num"] ="Primary"

Replace12 = Adult.loc[:,"education-num"] == 12
Adult.loc[Replace12, "education-num"] ="Primary"

Replace13 = Adult.loc[:,"education-num"] == 13
Adult.loc[Replace13, "education-num"] ="Secondary"

Replace14 = Adult.loc[:,"education-num"] == 14
Adult.loc[Replace14, "education-num"] ="Secondary"

Replace15 = Adult.loc[:,"education-num"] == 15
Adult.loc[Replace15, "education-num"] ="Secondary"

Replace16 = Adult.loc[:,"education-num"] == 16
Adult.loc[Replace16, "education-num"] ="Secondary"

new_education_unique = Adult.loc[:, "education-num"].unique()

print(new_education_unique)

#------------------------------------------------------------------------------
"Imputing missing values"

# Get the counts for each value
education_counts = Adult.loc[:,"education-num"].value_counts()

print(education_counts)

#define missing value
MissingValue = Adult.loc[:, "education-num"] == "?"

print(MissingValue)

#in this case there are 0 missing values, but if there were missing values
#I would replace with the highest count, I will still display the code

# Impute missing values
Adult.loc[MissingValue, "education-num"] = "Primary"

new_education_counts = Adult.loc[:,"education-num"].value_counts()

print(new_education_counts)

#------------------------------------------------------------------------------
"Consolidating categories if applicable"

### in this case because there are only 3 categories I do not feel there is any
## value gained from consolidating further

#------------------------------------------------------------------------------
""""One-hot encoding (dummy variables) for a categorical column with more than 
2 categories"""

# Create 3 new columns, one for each state in "education-num"
Adult.loc[:, "elementary"] = (Adult.loc[:, "education-num"] == "Elementary").astype(int)
Adult.loc[:, "primary"] = (Adult.loc[:, "education-num"] == "Primary").astype(int)
Adult.loc[:, "secondary"] = (Adult.loc[:, "education-num"] == "Secondary").astype(int)

print(Adult)

#------------------------------------------------------------------------------
"New columns created, obsolete deleted if applicable"

# Remove obsolete column
Adult = Adult.drop("education-num", axis=1)

print(Adult)

#------------------------------------------------------------------------------
"Plots for 1 or more categories"

## Scatter plot of capital gain vs elementary

elementary_scatter_plot = plt.scatter(Adult.loc[:,"elementary"],Adult.loc[:,"capital-gain"]) 

plt.show(elementary_scatter_plot)

## Scatter plot of age vs secondary

secondary_scatter_plot = plt.scatter(Adult.loc[:,"secondary"],Adult.loc[:,"age"]) 

plt.show(secondary_scatter_plot)

#------------------------------------------------------------------------------
"""Summary comment block on how the categorical data has been treated: decoded, 
imputed, consolidated, dummy variables created."""

##summary comment

"""I treated the education-num column by first converting it to an object data 
type, then decoded education levels 1-6 into the category "elementary",
education levels 7-12 into the category "primary" and education levels 13-16
into the cateogory "secondary". This column did not have any missing values 
so there was no data to inpute. If there were I would have replaced it with
the most frequent value. I did not consolidate further as there were only 
3 categories. I then used one hot encoding at created 3 dummy variables which
ultimately replaces the education-num column"""