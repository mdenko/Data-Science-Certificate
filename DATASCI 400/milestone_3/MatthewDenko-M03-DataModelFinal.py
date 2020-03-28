#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 11:56:48 2018

@author: matt.denko
"""
#------------------------------------------------------------------------------
#Part 1: Preparation of Data Set
"""Source citation for your data set
Data read from an easily and freely accessible source
Number of observations and attributes
Data types
Distribution of numerical variables
Distribution of categorical variables
A comment on each attribute
Removing cases with missing data
Removing outliers
Imputing missing values
Decoding
Consolidation
One-hot encoding
Normalization"""

#Part 2: Unsupervised Learning
"""Perform a K-Means with sklearn using some of your attributes.
Include at least one categorical column and one numeric attribute. 
Neither may be a proxy for the expert label in supervised learning.
Normalize the attributes prior to K-Means or justify why you didn't normalize.
Add the cluster label to the data set to be used in supervised learning"""

#Part 3: Unsupervised Learning
"""Ask a binary-choice question that describes your classification. 
Write the question as a comment.
Split your data set into training and testing sets using the proper function in sklearn.
Use sklearn to train two classifiers on your training set, like logistic regression and random forest. 
Apply your (trained) classifiers to the test set.
Create and present a confusion matrix for each classifier. 
Specify and justify your choice of probability threshold.
For each classifier, create and present 2 accuracy metrics based on the confusion matrix of the classifier.
For each classifier, calculate the ROC curve and it's AUC using sklearn. 
Present the ROC curve. Present the AUC in the ROC's plot."""

#------------------------------------------------------------------------------
"importing packages"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn.ensemble import RandomForestClassifier

#------------------------------------------------------------------------------
"defining functions"

##k-means
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
    
#------------------------------------------------------------------------------
"Source citation for your data set"

source_citation = "Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science."
print(source_citation)

#------------------------------------------------------------------------------
"Data read from an easily and freely accessible source"

##Reading url
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
Adult= pd.read_csv(url, header=None)
print(Adult)

##Assigning reasonable column names
Adult.columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
                 "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country",">50K, <=50k"]
print(Adult.columns)

#------------------------------------------------------------------------------
"Number of observations and attributes"

## Number of Observations
count_row = Adult.shape[0]
print(count_row)

## Number of Columns
count_col = Adult.shape[1]
print(count_col)

#------------------------------------------------------------------------------
"Data types"

##checking data types
adult_data_types = Adult.dtypes
print(adult_data_types)

#------------------------------------------------------------------------------
"Distribution of numerical variables & A comment on each attribute"

#Age
age_hist = plt.hist(Adult.loc[:,'age'])
plt.title("Age Histogram")
plt.xlabel('age')
plt.ylabel('frequency')
plt.show(age_hist)
age_comment = """Age is strongly skewed right and does not represent a 
normal distribution, there is a higher concentrate of younger participants to 
older."""
print(age_comment)

#fnlwgt
fnlwgt_hist = plt.hist(Adult.loc[:,'fnlwgt'])
plt.title("Fnlwg Histogram")
plt.xlabel('fnlwg')
plt.ylabel('frequency')
plt.show(fnlwgt_hist)
fnlwg_comment = """fnlwgt is also strongly right skewed. represents final weigh
which is the number of units in the target population that the responding unit
represents"""
print(fnlwg_comment)

#education-num
education_num_hist = plt.hist(Adult.loc[:,'education-num'])
plt.title("Education Num Histogram")
plt.xlabel('education-num')
plt.ylabel('frequency')
plt.show(education_num_hist)
education_num_comment = """education num has a somewhat bi-modal distribution
with one center around 8-12 and another at 14"""
print(education_num_comment)

#capital-gain
capital_gain_hist = plt.hist(Adult.loc[:,'capital-gain'])
plt.title("Capital Gain Histogram")
plt.xlabel('capital-gain')
plt.ylabel('frequency')
plt.show(capital_gain_hist)
capital_gain_comment = """capital gain is a single modal distribution that
appears slightly right skewed"""
print(capital_gain_comment)

#capital-loss
capital_loss_hist = plt.hist(Adult.loc[:,'capital-loss'])
plt.title("Capital Loss Histogram")
plt.xlabel('capital-loss')
plt.ylabel('frequency')
plt.show(capital_loss_hist)
capital_loss_comment = """captial loss is a single modal distribution that has
some skewed right outliers"""
print(capital_loss_comment)

#hours-per-week
hours_per_week_hist = plt.hist(Adult.loc[:,'hours-per-week'])
plt.title("Hours-Per-Week Histogram")
plt.xlabel('hours-per-week')
plt.ylabel('frequency')
plt.show(hours_per_week_hist)
hours_per_week_comment = """hours per week appears to be close to a normal
distribution, with some slight right skewness"""

#------------------------------------------------------------------------------
"Distribution of categorical variables & A comment on each attribute"

##to show the distribution of categorical data, I will create a bar chart of
#value counts

#workclass
workclass_dist = Adult['workclass'].value_counts().plot(kind='bar', title = "workclass value counts")
plt.show(workclass_dist)
workclass_comment = """workclass is heavily dominant in the private sector
with the remaining categories having a near equal distribution"""
print(workclass_comment)

#education
education_dist = Adult['education'].value_counts().plot(kind='bar', title = "education value counts")
plt.show(education_dist)
education_comment = """Education is heavily concentration in HS Grad,
some college, and Bachelors"""
print(education_comment)

#marital-status
marital_dist = Adult['marital-status'].value_counts().plot(kind='bar', title = "marital status value counts")
plt.show(marital_dist)
marital_comment = """marital status is heavily concentrated in Married, Never
Married, and Divorced. The remaining categories have a low concentration"""
print(marital_comment)

#occupation
occupation_dist = Adult['occupation'].value_counts().plot(kind='bar', title = "occupation value counts")
plt.show(occupation_dist)
occupation_comment = """Occupation has a fairly even distribution of values,
a few outliers such as private house service and armed forces are very lowly
concentrated"""
print(occupation_comment)

#relationship
relationship_dist = Adult['relationship'].value_counts().plot(kind='bar', title = "relationship value counts")
plt.show(relationship_dist)
relationship_comment = """relationship is highly concentrated in Husband and
not in family, and own child, with the remaining categores having lower
concentration"""
print(relationship_comment)

# race
race_dist = Adult['race'].value_counts().plot(kind='bar', title = "race value counts")
plt.show(race_dist)
race_comment = """Race is heavily dominant in white"""
print(race_comment)

# sex
sex_dist = Adult['sex'].value_counts().plot(kind='bar', title = "sex value counts")
plt.show(sex_dist)
sex_comment = """Sex has a higher concentration of males to females"""
print(sex_comment)

# native-country
native_dist = Adult['native-country'].value_counts().plot(kind='bar', title = "native country value counts")
plt.show(native_dist)
native_comment = """native country is heavily dominant to United States"""
print(native_comment)

# >50K, <=50k
fiftyk_dist = Adult['>50K, <=50k'].value_counts().plot(kind='bar', title = ">50K, <=50K value counts")
plt.show(fiftyk_dist)
fiftyk_comment = """There is a higher concentration of participants that make 
greater than 50k income then participants who make less than 50k income"""
print(fiftyk_comment)

#------------------------------------------------------------------------------
"Removing cases with missing data"

Adult = Adult.replace(to_replace= "?", value=float("NaN"))

#suming Nans
Adult_null = Adult.isnull().sum()
print(Adult_null)
print("""There are 0 columns with missing data, Below is the code I would 
      execute if there were:
Adult.loc[HasNan, "age"] =  np.nanmedian(Heart.loc[:,"age"])""")

#------------------------------------------------------------------------------
"Removing outliers"

##based off the distribution of age I will replace all values that are >55 with the median

# Replace outlier with median
AgeTooHigh = Adult.loc[:, "age"] > 55
print(AgeTooHigh)
Adult.loc[AgeTooHigh, "age"] = np.median(Adult.loc[:,"age"])
age_hist_new = plt.hist(Adult.loc[:,'age'])
plt.title("Age Histogram with Outliers Removed")
plt.xlabel('age')
plt.ylabel('frequency')
plt.show(age_hist_new)

##based off the distribution of captial-loss I will remove all values that are
# >1000 with the median

# Replace the outlier with median
CapTooHigh = Adult.loc[:, "capital-loss"] > 1000
print(CapTooHigh)
Adult.loc[CapTooHigh, "capital-loss"] = np.median(Adult.loc[:,"capital-loss"])
capital_loss_hist_new = plt.hist(Adult.loc[:,'capital-loss'])
plt.title("Capital Loss Histogram with Outliers Removed")
plt.xlabel('capital-loss')
plt.ylabel('frequency')
plt.show(capital_loss_hist_new)

#------------------------------------------------------------------------------
"Imputing missing values"

print("there are no missing values to impute")

#------------------------------------------------------------------------------
"Decoding"

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

new_education_num_unique = Adult.loc[:, "education-num"].unique()

print(new_education_num_unique)

#------------------------------------------------------------------------------
"Consolidation"

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
"One-hot encoding"

### Here is an example of one-hot encoding
## I will use a different encoder to create new columns for my model

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Adult.loc[:, "education-num"])
print(integer_encoded)

# onehot encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print(onehot_encoded)

### Adding encoding to the datase

# Create 3 new columns, one for each state in "education-num"
Adult.loc[:, "elementary"] = (Adult.loc[:, "education-num"] == "Elementary").astype(int)
Adult.loc[:, "primary"] = (Adult.loc[:, "education-num"] == "Primary").astype(int)
Adult.loc[:, "secondary"] = (Adult.loc[:, "education-num"] == "Secondary").astype(int)

# Create 1 new dummy column for >50K,<50K
Adult.loc[:, ">50K"] = (Adult.loc[:, ">50K, <=50k"] == ' >50K').astype(int)

# Removing obsolete columns
Adult = Adult.drop("education-num", axis=1)
Adult = Adult.drop(">50K, <=50k", axis=1)

#------------------------------------------------------------------------------
"Normalization"

#Extracting the numeric columns which make sense to normalize
age = Adult.loc[:,'age']
fnlwgt = Adult.loc[:,'fnlwgt']
capital_gain = Adult.loc[:,'capital-gain']
capital_loss = Adult.loc[:,'capital-loss']

# Normalizing numeric variables using numpy and z normalization
age_zscaled = (age - np.mean(age))/np.std(age)
fnlwgt_zscaled = (age - np.mean(fnlwgt))/np.std(fnlwgt)
capital_gain_zscaled = (capital_gain - np.mean(capital_gain))/np.std(capital_gain)
capital_loss_zscaled = (capital_loss - np.mean(capital_loss))/np.std(capital_loss)

#replacing the numeric values with the normalized values

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
"Unsupervised Learning: Perform a K-Means with sklearn using some of your attributes."

### I want to view relationship between the presence of secondary education
## and the captial gain recieved

##I will not normalize the columns because secondary is a dummy variable and 
#capital-gain has already been normalized

#extracting the columns
secondary =  Adult.loc[:,"secondary"]
capital_gain = Adult.loc[:,"capital-gain"]

#creating the dataframe
kmeansdf = pd.DataFrame()
kmeansdf.loc[:,0] = secondary
kmeansdf.loc[:,1] = capital_gain

#Centroid Guesses
ClusterCentroidGuesses = pd.DataFrame()
ClusterCentroidGuesses.loc[:,0] = [-1,1]
ClusterCentroidGuesses.loc[:,1] = [-1,1]

#Doing the clustering
kmeans = KMeans(n_clusters=2, init=ClusterCentroidGuesses, n_init=1).fit(kmeansdf)
Labels = kmeans.labels_
ClusterCentroids = pd.DataFrame(kmeans.cluster_centers_)
Plot2DKMeans(kmeansdf, Labels, ClusterCentroids, 'my cluster of secondary vs capital-gain')

#Adding the Label to the model
Adult.loc[:,"cluster_label"] = Labels

#------------------------------------------------------------------------------
"""Ask a binary-choice question that describes your classification. Write it as
a comment"""

#Question: Is education level in the presence of age, and capital gains a good 
#indicator of whether a participant makes >$50k annual salary?

#My expert label will be >50k

#------------------------------------------------------------------------------
"Split your data into training and test sets using the proper function in sklearn"

#subsetting dataset for non object columns
Adult_Data = pd.DataFrame()
Adult_Data.loc[:,"age"] = Adult.loc[:,"age"]
Adult_Data.loc[:,"fnlwgt"] = Adult.loc[:,"fnlwgt"]
Adult_Data.loc[:,"capital-gain"] = Adult.loc[:,"capital-gain"]
Adult_Data.loc[:,"capital-loss"] = Adult.loc[:,"capital-loss"]
Adult_Data.loc[:,"elementary"] = Adult.loc[:,"elementary"]
Adult_Data.loc[:,"primary"] = Adult.loc[:,"primary"]
Adult_Data.loc[:,"secondary"] = Adult.loc[:,"secondary"]
Adult_Data.loc[:,"cluster_label"] = Adult.loc[:,"cluster_label"]
Adult_Data.loc[:,">50K"] = Adult.loc[:,">50K"]

#Training = X
X = []

#Test = Y
Y = []

#splitting data into test and training sets using sklearn
X, Y = train_test_split(Adult_Data,test_size = .20)

print(X,"This is the Training Set")
print(Y,"This is the testing Set")

#------------------------------------------------------------------------------
"Classifier 1 - logistic regression"
#------------------------------------------------------------------------------
"Use sklearn to train two classifiers on your training set"
"Apply your (trained) classifiers to the test set."

#Creating the classifier
print ('\n Use logistic regression to predict >50K from other variables in Adult')
Target = ">50K"
Inputs = list(Adult_Data.columns)
Inputs.remove(Target)
clf = LogisticRegression()
clf.fit(X.loc[:,Inputs], X.loc[:,Target])
BothProbabilities = clf.predict_proba(Y.loc[:,Inputs])
probabilities = BothProbabilities[:,1]

#------------------------------------------------------------------------------
"Create and present a confusion matrix for each classifier."
"Specify and justify your choice of probability threshold."

# I will use a probability threshold of .5 in order to have a balance of 
#precision vs recall. 

print ('\nConfusion Matrix and Metrics')
Threshold = 0.5 # Some number between 0 and 1
print ("Probability Threshold is chosen to be:", Threshold)
predictions = (probabilities > Threshold).astype(int)
CM = confusion_matrix(Y.loc[:,Target], predictions)
tn, fp, fn, tp = CM.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)

#------------------------------------------------------------------------------
"Create and present 2 accuracy metrics based on the confusion matrix of the classifier."

AR = accuracy_score(Y.loc[:,Target], predictions)
print ("Accuracy rate:", np.round(AR, 2))
P = precision_score(Y.loc[:,Target], predictions)
print ("Precision:", np.round(P, 2))
R = recall_score(Y.loc[:,Target], predictions)
print ("Recall:", np.round(R, 2))

#------------------------------------------------------------------------------
"Calculate the ROC curve and it's AUC using sklearn"
"Present the ROC curve. Present the AUC in the ROC's plot."

# Creating False Positive Rate, True Posisive Rate, and probability thresholds
fpr, tpr, th = roc_curve(Y.loc[:,Target], probabilities)

#Calculating ROC
AUC = auc(fpr, tpr)

# Plotting the ROC Curve, presenting AUC in the plot
plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()

#------------------------------------------------------------------------------
"Classifier 2 - random forest"
#------------------------------------------------------------------------------
"Use sklearn to train two classifiers on your training set"

# creating random forest classifier
clf_rf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training inputs and learn how they relate
# to the training target
clf_rf.fit(X.loc[:,Inputs], X.loc[:,Target])

#------------------------------------------------------------------------------
"Apply your (trained) classifiers to the test set."

##Creating probabilities off of the test set
rf_probabilities = clf_rf.predict_proba(Y.loc[:,Inputs])[:,0]

#------------------------------------------------------------------------------
"Create and present a confusion matrix for each classifier."
"Specify and justify your choice of probability threshold."

# I will use a probability threshold of .5 in order to have a balance of 
#precision vs recall. 

print ('\nConfusion Matrix and Metrics')
rf_Threshold = 0.5 # Some number between 0 and 1
print ("Probability Threshold is chosen to be:", rf_Threshold)
rf_predictions = (rf_probabilities > rf_Threshold).astype(int)
rf_CM = confusion_matrix(Y.loc[:,Target], rf_predictions)
tn, fp, fn, tp = CM.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)

#------------------------------------------------------------------------------
"Create and present 2 accuracy metrics based on the confusion matrix of the classifier."

rf_AR = accuracy_score(Y.loc[:,Target], rf_predictions)
print ("Accuracy rate:", np.round(rf_AR, 2))
rf_P = precision_score(Y.loc[:,Target], rf_predictions)
print ("Precision:", np.round(rf_P, 2))
rf_R = recall_score(Y.loc[:,Target], rf_predictions)
print ("Recall:", np.round(rf_R, 2))

#------------------------------------------------------------------------------
"Calculate the ROC curve and it's AUC using sklearn"
"Present the ROC curve. Present the AUC in the ROC's plot."

# Creating False Positive Rate, True Posisive Rate, and probability thresholds
fpr, tpr, th = roc_curve(Y.loc[:,Target], rf_probabilities)

#Calculating ROC
rf_AUC = auc(fpr, tpr)

# Plotting the ROC Curve, presenting AUC in the plot
plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % rf_AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()

#------------------------------------------------------------------------------
print("""Conclusion: 
    
Data Preparation:
    
I read in the Adult data source which classifies if income exceeds 55k based
on census data. I assigned reasonable column names based on the website description.
I checked the distribution of each numeric column and categorical column and
gave a description of each attribute. I removed outliers from age and captial
#loss. I then checked to see if there were any missing numeric columns
 which there were not. I normalized the each of the 
numeric columns outside of education-num. I decoded education num into 
Secondary, Primary, and Elementary. I later encoded this column and replaced it
with three indicator columns. I also encoded the >50k column to create
a dummy variable for further analysis. I then consolidated the education column 
into 4 categories (Elementary, Middle-School,High-School, and College).

Unsupervised Learning: 
I created a cluster between the presence of secondary education
and the captial gain recieved. I then added the labels from this cluster to my
supervised model.

Supervised Learning:
The binary question that I asked is Is education level in the presence of
age, and capital gains a good indicator of whether a participant makes >$50k 
annual salary? I then split the dataset into a training set and testing set. I
first trained a logistic regression model on my training set and then applied
this classifier to my test set. I chose a threshold of .50 for both this
classifier and my later classifier because I wanted to balance the importance
of precision and recall. The result of the first classifier was a precesion
score of .69 and a recall .37 along with an accuracy score of .81. Which means
that the classifer got a fairly high proportion of correct predictions to total
predictions and thge proportion of predictions for a given class was relatively
high. However, the low recall score is concerning and means that the net
prediction potential of my model is not very high. The then trained a random
forest model on my training set and then applied the classifer to my test set,
using the same threshold as the first classification.The result of the second 
classifier was an accuracy rate of .18, a precision of .17 and a recall of .61.
This classification had a very low rate of correct predictions compared to
total predictions and the reliability of the predictions was very low. However,
the high recall score indicates that the net prediction potential of this model
is higher than the previous model. However, neither classification produced
encouraging results.""")


