import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("salary.csv")
#print(df)

x = df.iloc[:,1:2]
y = df.iloc[:,2:]

#print(x)
#print(y)

# Linear  Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x.values,y.values)

#print(lin_reg.predict(x.values))

linPredX = plt.figure(figsize = (10,7))
plt.scatter(x.values,y.values, color = 'red')
#print(type(x))
#print(type(lin_reg.predict(x.values)))
plt.plot(x.values,lin_reg.predict(x.values), color = 'blue')
#plt.savefig("linPredX.png")

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # setting degree can help to match data better
x_poly = poly_reg.fit_transform(x.values)
#print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

reg2 = plt.figure(figsize = (10,7))
plt.scatter(x.values,y.values)
# turn into poly values
plt.plot(x.values, lin_reg2.predict(poly_reg.fit_transform(x.values)))
#plt.savefig("reg2.png")

# prediction
print(lin_reg.predict([[6.6]]))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
