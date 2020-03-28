#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:47:24 2018

@author: matt.denko
"""
#-----------------------week 9 assignment instructions--------------------------

"""Build on your previous model-building Python script by adding the following:

1. Accuracy measures for your predicted values

a. Confusion Matrix (specify the probability threshold)
b. ROC Analysis with AUC score

2. Comments explaining the code blocks"""

#------------------------------------------------------------------------------
"""Short narrative on the data preparation for your chosen data set from Milestone 2
Import statements for libraries and your data set
Show data preparation.  Normalize some numeric columns, one-hot encode some categorical columns with 3 or more categories, remove or replace missing values, remove or replace some outliers.
Specify an appropriate column as your expert label for a classification.  (include decision comments)
K-Means based on some of your columns, but excluding the expert label.  Add the cluster labels to your dataset.
Split the data set into training and testing sets (include decision comments)
Create a classification model for the expert label (include decision comments)
Write out to a csv a dataframe of predicted and actual values 
Determine accuracy, which is the number of correct predictions divided by the total number of predictions (include brief preliminary analysis commentary)
Comments explaining the code blocks. 
The grader must be able to execute your script on their computer using only the run file (F5) button in spyder"""
#------------------------------------------------------------------------------

"""Short narrative on the data preparation for your chosen data set from Milestone 2
Import statements for libraries and your data set"
Show data preparation.  Normalize some numeric columns, one-hot encode some 
categorical columns with 3 or more categories, remove or 
replace missing values, remove or replace some outliers."""

#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *


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

## I want to one hot encode education to create 4 new columns

# Create 4 new columns, one for each state in "education"
Adult.loc[:, "elemenatry"] = (Adult.loc[:, "education"] == "Elementary").astype(int)
Adult.loc[:, "middle-school"] = (Adult.loc[:, "education"] == "Middle-School").astype(int)
Adult.loc[:, "high-school"] = (Adult.loc[:, "education"] == "High-School").astype(int)
Adult.loc[:, "college"] = (Adult.loc[:, "education"] == "College").astype(int)

print(Adult)

## I also want to one hot encode >50K,=<50K

# Create one new column
Adult.loc[:, ">50K"] = (Adult.loc[:, ">50K, <=50k"] == ' >50K').astype(int)

print(Adult)

#------------------------------------------------------------------------------
"Remove obsolete columns."

### The column education is no longer needed so I will remove it

# Remove obsolete column

Adult = Adult.drop("education", axis=1)

print(Adult)

#------------------------------------------------------------------------------
"Short Narrative"

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

#------------------------------------------------------------------------------
"""Specify an appropriate column as your expert label for a classification.  (include decision comments)
K-Means based on some of your columns, but excluding the expert label. 
Split the data set into training and testing sets (include decision comments)
Create a classification model for the expert label (include decision comments)
Write out to a csv a dataframe of predicted and actual values 
Determine accuracy, which is the number of correct predictions divided by the total number of predictions (include brief preliminary analysis commentary)
Comments explaining the code blocks."""

#------------------------------------------------------------------------------
"Specify an appropriate column as your expert label for a classification. " 
"(include decision comments)"

###The column that I will choose as the expert label for a classfication is
## >50k, <=50k. I am using this column because it is an indicator of whether or
#not a person makes greater than 50k. I want to predict a classification for 
#whether or not a person makes >50k.

#------------------------------------------------------------------------------
"K-Means based on some of your columns, but excluding the expert label. "

#Extracting and normalizing the columns"

Age =  Adult.loc[:,"capital-gain"]
Edu = Adult.loc[:,"education-num"]

Age_zscaled = (Age - np.mean(Age))/np.std(Age)

Edu_zscaled = np.array(Edu - np.mean(Edu))/np.std(Edu)

x = pd.DataFrame()

x.loc[:,0] = Age_zscaled
x.loc[:,1] = Edu_zscaled

ClusterCentroidGuesses = pd.DataFrame()
ClusterCentroidGuesses.loc[:,0] = [-1,1]
ClusterCentroidGuesses.loc[:,1] = [-1,1]


def Plot2DKMeans(Points, Labels, ClusterCentroids, Title):
    for LabelNumber in range(max(Labels)+1):
        LabelFlag = Labels == LabelNumber
        color =  ['c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y'][LabelNumber]
        marker = ['s', 'o', 'v', '^', '<', '>', '8', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'][LabelNumber]
        plt.scatter(Points.loc[LabelFlag,0], Points.loc[LabelFlag,1],
                    s= 100, c=color, edgecolors="black", alpha=0.3, marker=marker)
        plt.scatter(ClusterCentroids.loc[LabelNumber,0], ClusterCentroids.loc[LabelNumber,1], s=200, c="black", marker=marker)
    plt.title(Title)
    plt.show()
    
"Doing the clustering"
kmeans = KMeans(n_clusters=2, init=ClusterCentroidGuesses, n_init=1).fit(x)
Labels = kmeans.labels_
ClusterCentroids = pd.DataFrame(kmeans.cluster_centers_)

Plot2DKMeans(x, Labels, ClusterCentroids, 'my cluster of capital gain vs education num')

#------------------------------------------------------------------------------
"Add the cluster labels to your dataset."

Adult.loc[:,"cluster_labels"] = Labels

#------------------------------------------------------------------------------
"Split the data set into training and testing sets (include decision comments)"

#subsetting dataset for numeric columns

Adult_Data = pd.DataFrame()

Adult_Data.loc[:,0] = Adult.loc[:,"age"]
Adult_Data.loc[:,1] = Adult.loc[:,"fnlwgt"]
Adult_Data.loc[:,2] = Adult.loc[:,"education-num"]
Adult_Data.loc[:,3] = Adult.loc[:,"capital-gain"]
Adult_Data.loc[:,4] = Adult.loc[:,"capital-loss"]
Adult_Data.loc[:,5] = Adult.loc[:,"elemenatry"]
Adult_Data.loc[:,6] = Adult.loc[:,"middle-school"]
Adult_Data.loc[:,7] = Adult.loc[:,"high-school"]
Adult_Data.loc[:,8] = Adult.loc[:,"college"]
Adult_Data.loc[:,9] = Adult.loc[:,">50K"]

adult_matrix = pd.DataFrame.as_matrix(Adult_Data)

#defining the functions


def split_dataset(data, r): # split a dataset in matrix format, using a given ratio for the testing set
	N = len(data)	
	X = []
	Y = []
	
	if r >= 1: 
		print ("Parameter r needs to be smaller than 1!")
		return
	elif r <= 0:
		print ("Parameter r needs to be larger than 0!")
		return

	n = int(round(N*r)) # number of elements in testing sample
	nt = N - n # number of elements in training sample
	ind = -np.ones(n,int) # indexes for testing sample
	R = np.random.randint(N) # some random index from the whole dataset
	
	for i in range(n):
		while R in ind: R = np.random.randint(N) # ensure that the random index hasn't been used before
		ind[i] = R

	ind_ = list(set(range(N)).difference(ind)) # remaining indexes	
	X = data[ind_,:-1] # training features
	XX = data[ind,:-1] # testing features
	Y = data[ind_,-1] # training targets
	YY = data[ind,-1] # testing targests
	return X, XX, Y, YY

#Setting up the data for classification problem 
r = 0.2 # ratio of test data over all data (this can be changed to any number between 0.0 and 1.0 (not inclusive)

X, XX, Y, YY = split_dataset(adult_matrix, r)

#------------------------------------------------------------------------------
"Create a classification model for the expert label (include decision comments)"

# Logistic regression classifier
print ('\n\n\nLogistic regression classifier\n')
C_parameter = 50. / len(X) # parameter for regularization of the model
class_parameter = 'ovr' # parameter for dealing with multiple classes
penalty_parameter = 'l1' # parameter for the optimizer (solver) in the function
solver_parameter = 'saga' # optimization system used
tolerance_parameter = 0.1 # termination parameter
#####################

#Training the Model
clf = LogisticRegression(C=C_parameter, multi_class=class_parameter, penalty=penalty_parameter, solver=solver_parameter, tol=tolerance_parameter)
clf.fit(X, Y) 
print ('coefficients:')
print (clf.coef_) # each row of this matrix corresponds to each one of the classes of the dataset
print ('intercept:')
print (clf.intercept_) # each element of this vector corresponds to each one of the classes of the dataset

# Apply the Model
print ('predictions for test set:')
print (clf.predict(XX))
print ('actual class values:')
print (YY)

#------------------------------------------------------------------------------
"Write out to a csv a dataframe of predicted and actual values"

#Creating a dataframe

predicted_values = clf.predict(XX)
actual_values = YY

values = pd.DataFrame()
values.loc[:,0] = predicted_values
values.loc[:,1] = actual_values

#Writing to csv, removing index

csv = values.to_csv("MatthewDenko-L08-values.csv",index = False)

#Location where your file is stored

File_Location = os.getcwd()

print(File_Location) 

#------------------------------------------------------------------------------
"""Determine accuracy, which is the number of correct predictions divided by 
the total number of predictions (include brief preliminary analysis commentary)"""

#Total predictions

"6512"

#Total correct predictions"

"5242"

#Accuray

Accuracy = 5242/6512

"""I used logistic regression to predict values and I compared this to 
the original values. The accuracy of the model is around 80%."""


#-----------------------week 9 assignment instructions--------------------------

"""Build on your previous model-building Python script by adding the following:

1. Accuracy measures for your predicted values

a. Confusion Matrix (specify the probability threshold)
b. ROC Analysis with AUC score

2. Comments explaining the code blocks"""

#------------------------------------------------------------------------------
"1. Accuracy measures for your predicted values"
"a. Confusion Matrix (specify the probability threshold)"

# Assigned variables

T = predicted_values
Y = actual_values


# Creating accuracy measures

CM = confusion_matrix(T, Y)
print ("\n\nConfusion matrix:\n", CM)
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(T, Y)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)
P = precision_score(T, Y)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(T, Y)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(T, Y)
print ("\nF1 score:", np.round(F1, 2))

"""My model has high recall and low precision which indicates that the reliability
of my predictions is not very high, however the net prediction potential is 
very high"""

#------------------------------------------------------------------------------
"1. Accuracy measures for your predicted values"
"b. ROC Analysis with AUC score"

# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

fpr, tpr, th = roc_curve(T, Y) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#####################

plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()
####################

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(T, Y), 2), "\n")


"""My predicted values have an AUC score of .76 which means that my predictions
are relatively accurate. A value of 1 would indicate that my predictions are
100% accurate while a value of 0 would indicate that they are 0% accurate."""