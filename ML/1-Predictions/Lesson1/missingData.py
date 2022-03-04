import numpy as np
import pandas as pd

df = pd.read_csv("eksikveriler.csv")
# print(df.head())

# Missing Data Handling
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

age = df.iloc[:,1:4].values
#print(age)
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
print(age)
