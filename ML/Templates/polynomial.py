# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data load
df = pd.read_csv("salary.csv")

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
poly_reg = PolynomialFeatures(degree = 2) # setting degree of polynom can help to match data better
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

## 4th degree polynomial regression
poly_reg3 = PolynomialFeatures(degree = 4) # setting degree of polynom can help to match data better
x_poly3 = poly_reg3.fit_transform(x)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3, y)

# Prediction
print(lin_reg.predict([[6.6]]))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

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
reg3 = plt.figure(figsize = (10,7))
plt.scatter(x,y)
# turn into poly values
plt.plot(x, lin_reg3.predict(poly_reg3.fit_transform(x)))
#plt.savefig("reg3.png")
