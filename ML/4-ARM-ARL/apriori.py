# APriori
# https://github.com/ymoch/apyori/blob/master/apyori.py
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data as DataFrame
df = pd.read_csv("basket.csv", header = None)

# Show how many columns in the list
# print(df.head(11))

# Empty list
t = []
# Make a list with each transaction. i is each row. list with 20 items
for i in range (0,7501): # Range is equal to our lines of csv file
    t.append([str(df.values[i,j]) for j in range(0,20)])

# Import apriori library
from apyori import apriori
# 1% support with 20% confidence lift = confidence / support, min lenght of the relation is 2
rules = apriori(t, min_support = 0.01, min_confidence = 0.2, min_lift = 3, min_length = 2)

print(list(rules))
