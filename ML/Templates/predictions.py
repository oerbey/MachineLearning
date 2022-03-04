# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

## Data Import
df = pd.read_csv("newSalary.csv")

## SLICING
# list
y = df.iloc[:,-1:].values # get salary column as a numpy
id = df.iloc[:,0:1]
lvl = df.iloc[:,2:-1] # get rest of the columns
x = pd.concat([id, lvl], axis = 1).values
jlvl = df.iloc[:,2:3].values

# DataFrame
salaryDF = df.iloc[:,-1:]
idDF = df.iloc[:,0:1]
lvlDF = df.iloc[:,2:-1]

infoDF = pd.concat([idDF, lvlDF], axis = 1)
#print(lvlDF)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(lvlDF, salaryDF, test_size = 0.33, random_state = 42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train.values.ravel())

y_pred = regressor.predict(x_test)
#print(y_test)
#print(y_pred)
#print(y_pred[10, 10, 100])

import statsmodels.api as sm

x_l = lvlDF.iloc[:,[0,1,2]].values
x_l = np.array(x_l, dtype = float)
model = sm.OLS(salaryDF, x_l).fit()
print("Multi-Linear Regression Predictions\nCEO Salary Prediction")
print(model.predict([10,10,100]))
print("Manager Salary Prediction")
print(model.predict([7,10,100]))
#print(model.summary())
# x_l = lvlDF.iloc[:,[0,2]].values
# x_l = np.array(x_l, dtype = float)
# model = sm.OLS(salaryDF, x_l).fit()
# #print(model.summary())
# x_l = lvlDF.iloc[:,[0]].values
# x_l = np.array(x_l, dtype = float)
# model = sm.OLS(salaryDF, x_l).fit()
# #print(model.summary())
# x_train, x_test, y_train, y_test = train_test_split(jlvl, salaryDF, test_size = 0.33, random_state = 42)
# regressor.fit(x_train, y_train.values.ravel())
# y_pred = regressor.predict(x_test)
# #print(model.predict([10]))

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) # Degree of the polynomial
x_poly = poly_reg.fit_transform(lvl)
pol_reg = LinearRegression()
pol_reg.fit(x_poly, y)

print("\nPolynomial Regression Prediction\nCEO Salary")
print(pol_reg.predict(poly_reg.fit_transform([[10, 10, 100]])))
print("Manager Salary")
print(pol_reg.predict(poly_reg.fit_transform([[7, 10, 100]])))


# SVR - Support Vector Regression
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_scaled = sc1.fit_transform(lvl) # it will get standard scaler and transform
sc2 = StandardScaler()
y_scaled = sc2.fit_transform(y)

from sklearn.svm import SVR # svm = support vector machine

svr_reg = SVR(kernel = 'poly') # radial basis function
svr_reg.fit(x_scaled, y_scaled.ravel()) # create relation between two values

print("\nSVR Regression Prediction\nCEO Salary")
print(svr_reg.predict([[10, 10, 100]]))
print("Manager Salary")
print(svr_reg.predict([[7, 10, 100]]))

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(x,y)

print("\nDecision Tree Prediction\nCEO Salary")
print(r_dt.predict([[10, 10 ,10 ,100]]))
print("Manager Salary")
print(r_dt.predict([[10, 7 ,10 ,100]]))


# RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
# number of estimator is how many decision tree will be
rf_reg = RandomForestRegressor(n_estimators = 15, random_state = 0)
rf_reg.fit(x, y.ravel())

print("\nRandom Forest Prediction\nCEO Salary")
print(rf_reg.predict([[10, 10 ,10 ,100]]))
print("Manager Salary")
print(rf_reg.predict([[7, 7 ,10 ,100]]))

print("\nR2Scores: ")
from sklearn.metrics import r2_score
print("Random Forest R2 Score")
print(r2_score(y, rf_reg.predict(x)))

print("\nDecision Tree R2 Score")
print(r2_score(y, r_dt.predict(x)))

print("\nSVR R2 Score")
print(r2_score(y_scaled, svr_reg.predict(x_scaled)))

print("\nPolynomial R2 Score")
print(r2_score(y, pol_reg.predict(poly_reg.fit_transform(lvl))))

print("\nMulti Linear Regression R2 Score")
print(r2_score(y, model.predict(lvlDF)))
