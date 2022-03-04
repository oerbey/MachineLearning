from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

print("Start")
time_start = datetime.now()

df = pd.read_excel("merc.xlsx") # read excel file and inset into DataFrame
#print(df.describe())

# newfig = plt.figure(figsize=(10,6))
# sbn.displot(df["price"])
# plt.savefig("newfig.png", dpi=300)
# print("Done!")

ninetyNineDf = df.sort_values("price", ascending = False).iloc[131:] # index located. It will start after 131st by descending
#ninetyNineDf.describe() # describe 99% data

df = ninetyNineDf

# remove 1970 from data frame
df = df[df.year != 1970] # remove 1970 year

# remove transmission column. Alternative: It is possible turn data into numbers for regression test
df = df.drop("transmission", axis = 1) # removes transmission column

# Determine X and Y. Y is the values that we want to find
y = df["price"].values # get prices and make it numpy array
x = df.drop("price", axis = 1) # get all except from price

# Test split
from sklearn.model_selection import train_test_split
# set test size 30% and random is a random number for spliting method
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Import scaler
from sklearn.preprocessing import MinMaxScaler
# set scaler
scaler = MinMaxScaler()
# set x train and x test
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# import tensorflow for model and layer creation
from tensorflow.python.keras.models import Sequential # creates model
from tensorflow.python.keras.layers import Dense # creates layers

# x_train.shape # it shows number of data and layers (number of columns)
#(9090, 5)

# Create the model
model = Sequential()

# add hidden layers
model.add(Dense(12,activation = "relu")) # hidden layers
model.add(Dense(12,activation = "relu"))
model.add(Dense(12,activation = "relu"))
model.add(Dense(12,activation = "relu"))

model.add(Dense(1))

model.compile(optimizer = "adam", loss = "mse")

# train the model
# validation process is done in the training.
# Batch size set to prevent overload
# Epochs how many times the process will be done
model.fit(x = x_train, y = y_train, validation_data = (x_test, y_test), batch_size = 250, epochs = 300)

lossData = pd.DataFrame(model.history.history)
print(lossData.head()) # shows loss data

ld = lossData.plot()
plt.savefig("ld.png", dpi=300)

from sklearn.metrics import mean_absolute_error, mean_absolute_error
predictData = model.predict(x_test) # this gives prediced data

#mean_absolute_error(y_test, predictData)

estfig = plt.figure(figsize = (10,7))
plt.scatter(y_test, predictData)
plt.plot(y_test,y_test,"r*-")
plt.savefig("estfig.png", dpi=300)

print("End")
time_end = datetime.now()
print(f'TOTAL TIME = {(time_end - time_start).seconds} seconds')
