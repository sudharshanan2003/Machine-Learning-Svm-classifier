#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 22:25:33 2023

@author: sudharshanansankar
"""
"""
Based on the age and estimated salary we want to classify
whether a person will purchase a product or no 
0 - person will not purchase a product
1 - person will purchase a product 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets

df = pd.read_csv("Social_Networks_Ads.csv")
df.head()

df.shape
X = df.iloc[:, [2,3]] # On basis of what quantites you want to predict, that's why x axis is independent . Here we are considering only age and estimated salary 
Y = df.iloc[:, 4] # what we want to predict,dependent quantites. Here it is Purchased .
X.head()
Y.head()

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

print("Training data : ",X_Train.shape)
print("Training data : ",X_Test.shape)

# Feature Scaling( To scale the data )

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_Train, Y_Train)

# Predicting the test set results
Y_Pred = classifier.predict(X_Test)
print(Y_Pred)   # We will get 100 different y values for 100 different x values .

from sklearn import metrics
print('Accuracy Score: with linear kernel')

print(metrics.accuracy_score(Y_Test,Y_Pred)) # Accuracy is the difference between actual and predicted y value . Here it is 90%

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf') #Kernel has been changed to radial basis function (RBF) 
classifier.fit(X_Train, Y_Train)

# Predicting the test set results
Y_Pred = classifier.predict(X_Test)

print('Accuracy Score: with default rbf kernel')
print(metrics.accuracy_score(Y_Test,Y_Pred))

from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', gamma = 10,  random_state=0)  # 93% accuracy
classifier = SVC(kernel = 'rbf', gamma = 15, C=7,  random_state=0) 
classifier.fit(X_Train, Y_Train)

# Predicting the test set results

Y_Pred = classifier.predict(X_Test)

print('Accuracy Score On Test Data: with default rbf kernel')
print(metrics.accuracy_score(Y_Test,Y_Pred))

svc=SVC(kernel='poly', degree = 4)
svc.fit(X_Train,Y_Train)

y_pred=svc.predict(X_Test)
print('Accuracy Score:with poly kernel and degree ')
print(metrics.accuracy_score(Y_Test,Y_Pred))  

import matplotlib.pyplot as plt

plt.scatter(X_Train[:, 0], X_Train[:, 1],c=Y_Train)  
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Training Data')
plt.show()

import matplotlib.pyplot as plt


plt.scatter(X_Test[:, 0], X_Test[:, 1],c=Y_Test)  
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Test Data')
plt.show()

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_Train, Y_Train)

# Predicting the test set results
Y_Pred = classifier.predict(X_Test)

# Plot data points

plt.scatter(X_Test[:, 0], X_Test[:, 1],c=Y_Test)  
#plt.scatter(X_Train[:, 0], X_Train[:, 1],c=Y_Train) 

# Create the hyperplane
w = classifier.coef_[0]
a = -w[0] / w[1] #slope of the line 
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (classifier.intercept_[0]) / w[1]  

# Plot the hyperplane
plt.plot(xx, yy)
plt.axis("off"), plt.show();

