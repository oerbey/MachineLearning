## TEST DATA PREPARATION
# 1. libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2. Data processing
# 2.1 Data import
df = pd.read_csv("eksikveriler.csv")
# print(df.head())

# 2.2 Missing Data Handling
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

age = df.iloc[:,1:4].values
#print(age)
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
#print(age)

# Encoder it gets categoric data -> numeric
# set country as an array
country = df.iloc[:,0:1].values
#print(country)

# Use preprocessing from sklearn
from sklearn import preprocessing

# Label Encoding: it coverts to number
le = preprocessing.LabelEncoder()

# fit ML learning process, ML apply process is transform
country[:,0] = le.fit_transform(df.iloc[:,0])

# One-hot Encoding (kolon basliklarini etiketlere tasir ve 1 veya 0 degeri atar)
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
#print(country)

# Numpay arrays -> data frames
# data frames have index column, arrays don't have index and column name.
# there are 21 rows and that's why range is 22
countryDf = pd.DataFrame(data = country, index = range(22), columns = ['fr', 'tr', 'us'])

identityDf = pd.DataFrame(data = age, index = range(22), columns = ['boy', 'kilo', 'yas'])

gender = df.iloc[:,-1].values

genderDf = pd.DataFrame(data = gender, index = range(22), columns = ['cinsiyet'])

# union data frames
ci = pd.concat([countryDf, identityDf], axis = 1)
# axis = 1 add columns near
cig = pd.concat([ci, genderDf], axis = 1)
#print(cig)

# Test data preparation and data split
from sklearn.model_selection import train_test_split # till some rows, its test and after it's train
x_train, x_test, y_train, y_test = train_test_split(ci, genderDf, test_size = 0.33, random_state = 42)
# x bagimsiz degisken, bagimli degisten/ hedef = cinsiyet
# kisin cinsiyetini tahmin etmede boy ve kilonun tekisi nedir?
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train) # it will get standard scaler and transform
X_test = sc.fit_transform(x_test)
# farkli veriler ortak bir duruma getiriliyor
print(X_train)
