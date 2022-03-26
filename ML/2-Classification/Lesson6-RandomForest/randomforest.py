# K Nearest Neighborhood

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/onurerbey/github/MachineLearning/ML/2-Classification/Lesson2-KNN/testdata.csv")
#print(df.head())
#print(df.isnull().sum().sum()) # sum of null values

# 2.2 Missing Data Handling
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

x = df.iloc[:,1:4] # dependent variable
y = df.iloc[:,4:] # independent variable

x = pd.DataFrame(imputer.fit_transform(x))


## Test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

## Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

xTrain = sc.fit_transform(x_train) # fit trains
xTest = sc.transform(x_test) # transorm applies/uses the data

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 42)
logreg.fit(xTrain, y_train.values.ravel())

y_pred = logreg.predict(xTest)
#print(y_pred)
#print(y_test)

## Confusion Matrix
# this shows how our data is classified. First row is correct classification
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)

# print("KNN\n")
## KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski')
KNN.fit(xTrain, y_train.values.ravel())

y_pred = KNN.predict(xTest)

#print(y_pred)
#print(y_test)

cm = confusion_matrix(y_test, y_pred) # shows correct/wrong predictions
# print(cm)

from sklearn.svm import SVC
svc = SVC(kernel = 'rbf') # rbf, sigmoid, linear, poly
svc.fit(xTrain, y_train.values.ravel())

y_pred = svc.predict(xTest)

#print(y_test)
#print(y_pred)
cm = confusion_matrix(y_test, y_pred)
#print(cm)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB() # create object
gnb.fit(xTrain, y_train.values.ravel())

y_pred = gnb.predict(xTest)

cm = confusion_matrix(y_test, y_pred)
#print(cm)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(xTrain, y_train.values.ravel())
y_pred = dtc.predict(xTest)

cm = confusion_matrix(y_test, y_pred)
#print('DTC')
#print(cm)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'gini') # entropy, gini
rfc.fit(xTrain, y_train.values.ravel())
y_pred = rfc.predict(xTest)

print(y_pred)
print(y_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
