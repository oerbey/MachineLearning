import numpy as np
import pandas as pd

df = pd.read_csv("eksikveriler.csv")

country = df.iloc[:,0:1].values
#print(country)

from sklearn import preprocessing

# Label Encoding
le = preprocessing.LabelEncoder()

# fit ML learning process, ML apply process is transform
country[:,0] = le.fit_transform(df.iloc[:,0])

# One hot Encoding
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)
