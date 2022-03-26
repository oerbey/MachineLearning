## Classification Homework with Iris Data Set
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
df = pd.read_csv('iris.csv')

#print(df.head())
# print(df.describe())
# print(df.isnull().sum())

## Determine dependent & independent variables
x = df.iloc[:,0:4] # dependent variable
y = df.iloc[:,4:] # independent variable

## Test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

## Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train) # fit trains
X_test = sc.transform(x_test) # transorm applies/uses the data

## Classifcation Algorithms start
## Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 42)
logreg.fit(X_train, y_train.values.ravel())

y_pred = logreg.predict(X_test)

## Confusion Matrix
# this shows how our data is classified. First row is correct classification
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print('CM for Logreg')
print(cm)
# print(y_pred)
# print(y_test)

## KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski') # euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis
KNN.fit(X_train, y_train.values.ravel())

y_pred = KNN.predict(X_test)

cm = confusion_matrix(y_test, y_pred) # shows correct/wrong predictions
print('\nCM for KNN')
print(cm)

## SVC (SVM Classifier)
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf') # rbf, sigmoid, linear, poly
svc.fit(X_train, y_train.values.ravel())

y_pred = svc.predict(X_test)

#print(y_test)
#print(y_pred)
cm = confusion_matrix(y_test, y_pred)
print('\nSVC CM')
print(cm)

## Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB() # create object
gnb.fit(X_train, y_train.values.ravel())

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('\nNaive Bayes CM')
print(cm)

## Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy') # gini, entropy
dtc.fit(X_train, y_train.values.ravel())
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('\nDTC CM')
print(cm)

## Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'gini') # entropy, gini
rfc.fit(X_train, y_train.values.ravel())
y_pred = rfc.predict(X_test)
y_proba = rfc.predict_proba(X_test)

cm = confusion_matrix(y_test, y_pred)
print('\nRF CM')
print(cm)
