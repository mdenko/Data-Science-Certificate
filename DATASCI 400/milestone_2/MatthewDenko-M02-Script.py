#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 10:00:25 2018

@author: matt.denko
"""

"""Read in the data from a freely available source on the internet.  
Account for outlier values in numeric columns (at least 1 column).
Replace missing numeric data (at least 1 column).
Normalize numeric values (at least 1 column, but be consistent with numeric data).
Bin numeric variables (at least 1 column).
Consolidate categorical data (at least 1 column).
One-hot encode categorical data with at least 3 categories (at least 1 column).
Remove obsolete columns.
Save your script as Studentname-M02-Script.py, and have it write your data as 
Studentname-M02-Dataset.csv (the dataset itself), replacing Studentname with 
your own. You do not need to submit your CSV file--I will run your script to produce it."""

#------------------------------------------------------------------------------
"import statements"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#------------------------------------------------------------------------------
"Read in the data from a freely available source on the internet."

##Reading url

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

Adult= pd.read_csv(url, header=None)

print(Adult)

##Assigning reasonable column names

Adult.columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
                 "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country",">50K, <=50k"]

print(Adult)

#------------------------------------------------------------------------------
"Account for outlier values in numeric columns (at least 1 column)."

##checking data types

adult_data_types = Adult.dtypes

print(adult_data_types)

##checking distribution for outliers in numeric columns

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

##based off the distribution of age I will replace all values that are >55 with the median

# Replace outlier with median

TooHigh = Adult.loc[:, "age"] > 55

print(TooHigh)

Adult.loc[TooHigh, "age"] = np.median(Adult.loc[:,"age"])

print(Adult["age"])

#------------------------------------------------------------------------------
"Replace missing numeric data (at least 1 column)."

#re-checking data types

adult_data_types = Adult.dtypes

print(adult_data_types)

#replacing ? wiht Nan for numeric columns

Adult = Adult.replace(to_replace= "?", value=float("NaN"))

print(Adult)

#suming Nans

Adult_null = Adult.isnull().sum()

print(Adult_null)

print("""There are 0 Nans for numeric columns, Below is the code I would execute if there were:

Adult.loc[HasNan, "age"] =  np.nanmedian(Heart.loc[:,"age"])""")

 #------------------------------------------------------------------------------
"Normalize numeric values (at least 1 column, but be consistent with numeric data)."

#re-checking data types

adult_data_types = Adult.dtypes

print(adult_data_types)

#Extracting the numeric columns which make sense to normalize
#Education-Num did not make sense to normalize as it is a code for a category

age = Adult.loc[:,'age']

fnlwgt = Adult.loc[:,'fnlwgt']

capital_gain = Adult.loc[:,'capital-gain']

capital_loss = Adult.loc[:,'capital-loss']

hours_per_week = Adult.loc[:,'hours-per-week']

# Normalizing numeric variables using numpy and z normalization

age_zscaled = (age - np.mean(age))/np.std(age)

fnlwgt_zscaled = (age - np.mean(fnlwgt))/np.std(fnlwgt)

capital_gain_zscaled = (capital_gain - np.mean(capital_gain))/np.std(capital_gain)

capital_loss_zscaled = (capital_loss - np.mean(capital_loss))/np.std(capital_loss)

hours_per_week_zscaled = (hours_per_week - np.mean(hours_per_week))/np.std(hours_per_week)

## Comparing the distribution of numeric columns before and after normalization

# age
age_hist = plt.hist(age)

plt.show(age_hist)

age_zscaled_hist = plt.hist(age_zscaled)

plt.show(age_zscaled_hist)

# fnlwgt

fnlwgt_hist = plt.hist(fnlwgt)

plt.show(fnlwgt_hist)

fnlwgt_zscaled_hist = plt.hist(fnlwgt_zscaled)

plt.show(fnlwgt_zscaled_hist)

# capital-gain

capital_gain_hist = plt.hist(capital_gain)

plt.show(capital_gain_hist)

capital_gain_zscaled_hist = plt.hist(capital_gain_zscaled)

plt.show(capital_gain_zscaled_hist)

# capital-loss

capital_loss_hist = plt.hist(capital_loss)

plt.show(capital_loss_hist)

capital_loss_zscaled_hist = plt.hist(capital_loss_zscaled)

plt.show(capital_loss_zscaled_hist)

# hours-per-week

hours_per_week_hist = plt.hist(hours_per_week)

plt.show(hours_per_week_hist)

hours_per_week_zscaled_hist = plt.hist(hours_per_week_zscaled)

plt.show(hours_per_week_zscaled_hist)

#replacing the numeric values with the normalized values, I will not replace
#hours-per-week as I will be binning that value

# age

replace_age = Adult.loc[:,"age"] = age_zscaled

print(Adult["age"])

# fnlwgt

replace_fnlwgt = Adult.loc[:,"fnlwgt"] = fnlwgt_zscaled

print(Adult["fnlwgt"])

# capital-gain

replace_capital_gain = Adult.loc[:,"capital-gain"] = capital_gain_zscaled

print(Adult["capital-gain"])

# capital-loss

replace_capital_loss = Adult.loc[:,"capital-loss"] = capital_loss_zscaled

print(Adult["capital-loss"])

#------------------------------------------------------------------------------
"Bin numeric variables (at least 1 column)."

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

#Replacing hours-per-week column with the Low, Medium, High Values

Replace_HPW = Adult.loc[:,"hours-per-week"] = Binned_EqW

print(Adult["hours-per-week"])

#------------------------------------------------------------------------------
"Consolidate categorical data (at least 1 column)."

#checking data types

adult_data_types = Adult.dtypes

print(adult_data_types)

### Check the unique values for all categorical data types

## workclass

workclass_unique = Adult.loc[:, "workclass"].unique()

print(workclass_unique)

## education

education_unique = Adult.loc[:,"education"].unique()

print(education_unique)

## marital-status

marital_unique = Adult.loc[:,"marital-status"].unique()

print(marital_unique)

##occupation

occupation_unique = Adult.loc[:,"occupation"].unique()

print(occupation_unique)

##relationship

relationship_unique = Adult.loc[:,"relationship"].unique()

print(relationship_unique)

##race

race_unique = Adult.loc[:,"race"].unique()

print(race_unique)

##sex

sex_unique = Adult.loc[:,"sex"].unique()

print(sex_unique)

##native-country

native_unique = Adult.loc[:,"native-country"].unique()

print(native_unique)

##>50K,<=50K

K_unique = Adult.loc[:,">50K, <=50k"].unique()

print(K_unique)

### I want to consolidsate Education into 4 different values, Elementary
## Middle-School, High-School, College

Replace = Adult.loc[:,"education"] == " Bachelors"
Adult.loc[Replace, "education"] = "College"

Replace2 = Adult.loc[:,"education"] == " HS-grad"
Adult.loc[Replace2, "education"] = "High-School"

Replace3 = Adult.loc[:,"education"] == " 11th"
Adult.loc[Replace3, "education"] = "High-School"

Replace4 = Adult.loc[:,"education"] == " Masters"
Adult.loc[Replace4, "education"] = "College"

Replace5 = Adult.loc[:,"education"] == " 9th"
Adult.loc[Replace5, "education"] = "High-School"

Replace6 = Adult.loc[:,"education"] == " Some-college"
Adult.loc[Replace6, "education"] = "College"

Replace7 = Adult.loc[:,"education"] == " Assoc-acdm"
Adult.loc[Replace7, "education"] = "College"

Replace8 = Adult.loc[:,"education"] == " 7th-8th"
Adult.loc[Replace8, "education"] = "Middle-School"

Replace9 = Adult.loc[:,"education"] == " Doctorate"
Adult.loc[Replace9, "education"] = "College"

Replace10 = Adult.loc[:,"education"] == " Prof-school"
Adult.loc[Replace10, "education"] = "College"

Replace11 = Adult.loc[:,"education"] == " 5th-6th"
Adult.loc[Replace11, "education"] = "Middle-School"

Replace12 = Adult.loc[:,"education"] == " 10th"
Adult.loc[Replace12, "education"] = "High-School"

Replace13 = Adult.loc[:,"education"] == " 1st-4th"
Adult.loc[Replace13, "education"] = "Elementary"

Replace14 = Adult.loc[:,"education"] == " Preschool"
Adult.loc[Replace14, "education"] = "Elementary"

Replace15 = Adult.loc[:,"education"] == " 12th"
Adult.loc[Replace15, "education"] = "High-School"

Replace16 = Adult.loc[:,"education"] == " Assoc-voc"
Adult.loc[Replace16, "education"] = "College"

new_education_unique = Adult.loc[:, "education"].unique()

print(new_education_unique)

#------------------------------------------------------------------------------
"One-hot encode categorical data with at least 3 categories (at least 1 column)"

## I want to one hot encode ecducation to create 4 new columns

# Create 4 new columns, one for each state in "education"
Adult.loc[:, "elemenatry"] = (Adult.loc[:, "education"] == "Elementary").astype(int)
Adult.loc[:, "middle-school"] = (Adult.loc[:, "education"] == "Middle-School").astype(int)
Adult.loc[:, "high-school"] = (Adult.loc[:, "education"] == "High-School").astype(int)
Adult.loc[:, "college"] = (Adult.loc[:, "education"] == "College").astype(int)

print(Adult)

#------------------------------------------------------------------------------
"Remove obsolete columns."

### The column education is no longer needed so I will remove it

# Remove obsolete column

Adult = Adult.drop("education", axis=1)

print(Adult)

#------------------------------------------------------------------------------
"Write your data as Studentname-M02-Dataset.csv (the dataset itself)"

#Writing to csv, removing index

csv = Adult.to_csv("MatthewDenko-M02-Dataset.csv",index = False)

#Location where your file is stored

File_Location = os.getcwd()

print(File_Location)

#------------------------------------------------------------------------------
"Summary"

"""I read in the Adult data source which clasifies if income exceeds 55k based
on census data. I assigned reasonable column names based on the website description.
I checked the distribution of each numeric column to identify outliers. The only
column that had values which needed to be replaced was Age. I replaced all values
greater than 55 with the median value of Age. I then checked to see if there 
were any missing numeric columsn which there were not. I normalized the each of the 
numeric columns outside of education-num. I replaced all but hours-per-week
as I was going to bin that column. I then consolidated the education column 
into 4 categories (Elementary, Middle-School,High-School, and College). 
I then one-hot encoded this column created 4 new columns and then removed the 
original education column. The data is outputted to a csv file which will be 
stored in your current working directory"""

