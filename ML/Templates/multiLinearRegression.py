import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/onurerbey/OneDrive/Python/ML/Lesson1/veriler.csv")

# print(df.describe())

# NaN values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
age = df.iloc[:,1:4].values
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])

# convert gender column to numbers
gender = df.iloc[:,-1:].values
country = df.iloc[:,0:1].values

#print(gender)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
gender[:,-1] = le.fit_transform(df.iloc[:,-1])
country[:,0] = le.fit_transform(df.iloc[:,0])
#print(gender)

ohe = preprocessing.OneHotEncoder()
gender = ohe.fit_transform(gender).toarray()
country = ohe.fit_transform(country).toarray()
#print(gender)

countryDf = pd.DataFrame(data = country, index = range(22), columns = ['fr', 'tr', 'us'])

identityDf = pd.DataFrame(data = age, index = range(22), columns = ['boy', 'kilo', 'yas'])

#gender = df.iloc[:,-1].values

genderDf = pd.DataFrame(data = gender[:,:1], index = range(22), columns = ['cinsiyet'])
#print(genderDf)
# union data frames
ci = pd.concat([countryDf, identityDf], axis = 1)
# axis = 1 add columns near
cig = pd.concat([ci, genderDf], axis = 1)
#print(cig)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(ci, genderDf, test_size = 0.33, random_state = 42)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

height = cig["boy"].values
#print(height)
left = cig.iloc[:,:3]
right = cig.iloc[:,4:]
#print(left)
#print(right)
newData = pd.concat([left, right], axis = 1) # removed height and created a new DataFrame
#print(newData)
#print(type(newData))

x_train, x_test, y_train, y_test = train_test_split(newData, height, test_size = .33, random_state = 0)
#print(x_train)

regressor2 = LinearRegression()
regressor2.fit(x_train, y_train)
y_pred = regressor2.predict(x_test)
#print(y_pred)
#print(y_test)

# this library for p vlaue
import statsmodels.api as sm
# add beta 0 values. We have 22 rows.
# Create an array with 22 values with 1 and 1 column as integer, appending series is newData and add as column
x = np.append(arr = np.ones((22,1)).astype(int), values = newData, axis = 1)
# print(x) # added beta 0 values

# Backward Elimination
# Create an array for each column in the data frame
x_l = newData.iloc[:,[0,1,2,3,4,5]].values # get all columns to calculate p values
x_l = np.array(x_l, dtype = float)
model = sm.OLS(height, x_l).fit()
#print(model.summary())

x_l = newData.iloc[:,[0,1,2,3,5]].values # get all columns to calculate p values
x_l = np.array(x_l, dtype = float)
model = sm.OLS(height, x_l).fit()
#print(model.summary())

x_l = newData.iloc[:,[0,1,2,3]].values # get all columns to calculate p values
x_l = np.array(x_l, dtype = float)
model = sm.OLS(height, x_l).fit()
print(model.summary())
# can perform regression test again after this
