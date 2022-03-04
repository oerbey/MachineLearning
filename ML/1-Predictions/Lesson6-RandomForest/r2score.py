# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data load
df = pd.read_csv("/Users/onurerbey/OneDrive/Python/ML/Lesson3/salary.csv")

# Slicing
x = df.iloc[:,1:2]
y = df.iloc[:,2:]

# Numpy array convert
x = x.values
y = y.values

# Linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Polynomial regression (non-linear model)
## 2nd degree polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # setting degree of polynom can help to match data better
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

## 4th degree polynomial regression
#poly_reg3 = PolynomialFeatures(degree = 4) # setting degree of polynom can help to match data better
#x_poly3 = poly_reg3.fit_transform(x)
#lin_reg3 = LinearRegression()
#lin_reg3.fit(x_poly3, y)

# Prediction
#print(lin_reg.predict([[6.6]]))
#print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

# Visualization
## Linear regression viualization
linPredX = plt.figure(figsize = (10,7))
plt.scatter(x,y, color = 'red')
#print(type(x))
#print(type(lin_reg.predict(x.values)))
plt.plot(x,lin_reg.predict(x), color = 'blue')
#plt.savefig("linPredX.png")

## 2nd degree Polynom regression visualization
reg2 = plt.figure(figsize = (10,7))
plt.scatter(x,y)
# turn into poly values
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(x)))
#plt.savefig("reg2.png")

## 4th degree Polynom regression visualization
#reg3 = plt.figure(figsize = (10,7))
#plt.scatter(x,y)
# turn into poly values
#plt.plot(x, lin_reg3.predict(poly_reg3.fit_transform(x)))
#plt.savefig("reg3.png")


# SVR - Support Vector Regression
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_scaled = sc1.fit_transform(x) # it will get standard scaler and transform
sc2 = StandardScaler()
y_scaled = sc2.fit_transform(y)

from sklearn.svm import SVR # svm = support vector machine

svr_reg = SVR(kernel = 'rbf') # radial basis function
svr_reg.fit(x_scaled, y_scaled) # create relation between two values

## Visualize
svrPlt = plt.figure(figsize = (10,7))
plt.scatter(x_scaled, y_scaled, color = 'red')
# find all x scaled values in svr regression
plt.plot(x_scaled, svr_reg.predict(x_scaled), color = 'blue')
#plt.savefig('svrPlt.png', dpi = 200)

#print(svr_reg.predict([[11]]))
#print(svr_reg.predict([[6.6]]))

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(x,y)

dtPlt = plt.figure(figsize = (10,7))
plt.scatter(x,y,color = 'red')
plt.plot(x, r_dt.predict(x), color = 'blue')
#plt.savefig('dtPlt.png')

## predict
# print(r_dt.predict([[11]]))
# print(r_dt.predict([[6.6]]))

# RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
# number of estimator is how many decision tree will be
rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_reg.fit(x, y.ravel())

#print(rf_reg.predict([[6.6]]))

rfPlt = plt.figure(figsize = (10,7))
plt.scatter(x,y,color = 'red')
plt.plot(x, rf_reg.predict(x), color = 'blue')
#plt.savefig('rfPlt.png')


# R SQUARE
from sklearn.metrics import r2_score
print("Random Forest R2 Score")
print(r2_score(y, rf_reg.predict(x)))

print("Decision Tree R2 Score")
print(r2_score(y, r_dt.predict(x)))

print("SVR R2 Score")
print(r2_score(y_scaled, svr_reg.predict(x_scaled)))

print("Polynomial R2 Score")
print(r2_score(y, lin_reg2.predict(poly_reg.fit_transform(x))))

print("Linear Regression R2 Score")
print(r2_score(y, lin_reg.predict(x)))
