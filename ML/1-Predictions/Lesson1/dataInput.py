# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data input
df = pd.read_csv("veriler.csv")

# Data processing
height = df[['boy']]
# print(height)

# Python Classes
class human:
    height = 180
    def run(self, b):
        return b + 10

Onur = human()
print(Onur.height)
print(Onur.run(90))

# Python lists

l = [1,2,3,4] # lists
