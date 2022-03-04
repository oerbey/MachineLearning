import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('tennis.csv')
#print(df.head())
#print(df.describe())
#print(df.isnull())

outlook = df.iloc[:,0:1].values
windy = df.iloc[:,3:4].values
play = df.iloc[:,-1:].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
outlook[:,0] = le.fit_transform(df.iloc[:,0])
windy = le.fit_transform(df["windy"])
play[:,-1] = le.fit_transform(df.iloc[:,-1])

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()

#print(outlook)
#print(windy)
#print(play)
outlookDF = pd.DataFrame(data = outlook, index = range(14), columns = ['sunny', 'overcast', 'rainy'])
#print(outlookDF)
weatherDF = pd.DataFrame(data = df, index = range(14), columns = ['temperature', 'humidity'])
#print(weatherDF)
windyDF = pd.DataFrame(data = windy, index = range(14), columns = ['windy'])
playDF = pd.DataFrame(data = play, index = range(14), columns = ['play'])
owDF = pd.concat([outlookDF, weatherDF], axis = 1)
#print(owDF)
owwDF = pd.concat([owDF, windyDF], axis = 1)
#print(type(windyDF))
#print(owwDF)
owwpDF = pd.concat([owwDF, playDF], axis = 1)
#print(owwpDF)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(owwDF, playDF, test_size = 0.33, random_state = 42)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
predList = []
for i in y_pred:
    if i <= 0.5:
        predList.append('no')
    else:
        predList.append('yes')
print(predList)
#print(y_pred) # 3 correct  prediction
print(y_test)
newData = pd.concat([owDF, playDF], axis = 1)
#print(newData)
import statsmodels.api as sm
x = np.append(arr = np.ones((14,1)).astype(int), values = newData, axis = 1)
#print(x)
x_l = newData.iloc[:,[0,1,2,3,4,5]].values # get all columns to calculate p values
x_l_array = np.array(x_l, dtype = float)
model = sm.OLS(np.asarray(weatherDF['temperature']), x_l_array).fit()
#print(model.summary())

x_l = newData.iloc[:,[0,1,2,3,5]].values # get all columns to calculate p values
x_l_array = np.array(x_l, dtype = float)
model = sm.OLS(np.asarray(weatherDF['temperature']), x_l_array).fit()
#print(model.summary())

x_l = newData.iloc[:,[1,2,3,5]].values # get all columns to calculate p values
x_l_array = np.array(x_l, dtype = float)
model = sm.OLS(np.asarray(weatherDF['temperature']), x_l_array).fit()
print(model.summary())
#print(x_train.iloc[:,[1,2,3,5]])
x_train = x_train.iloc[:,[1,2,3,5]]
x_test = x_test.iloc[:,[1,2,3,5]]

regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
#print(y_pred) # 4 correct prediction
predList2 = []
for n in y_pred:
    if n <= 0.5:
        predList2.append('no')
    else:
        predList2.append('yes')
print(predList2)
