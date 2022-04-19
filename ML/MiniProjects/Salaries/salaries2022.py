# Developer salaries in Turkey for 2022
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("maas-anketi.csv")

# print(df["level"].value_counts())

position = df.iloc[:,1:2]
salary = df.iloc[:,-1:]
level = df.iloc[:,0:1]

# convert salary DF to list & get min salary from the salary range as DF
minSalaryList = salary.values.tolist()
minSalary = []

for i in range(0, 5030):
    y = str(minSalaryList[i]).split(" ")
    m = y[0].split("'")
    try:
        n = m[1]
        minSalary.append(n)
        continue
    except:
        minSalary.append('No Data')
        continue

minSalaryDF = pd.DataFrame(minSalary, columns=["salaries"])

# Create new DF with position and minSalary for each person
salPosDf = pd.concat([level, position, minSalaryDF], axis = 1)

# Remove lines where salary is not available
salPosDf = salPosDf[salPosDf.salaries != 'No Data']

## Export new DF as CSV
salPosDfCsv = salPosDf.to_csv('levPosSal.csv', index = True)

def dfCsv():
    return salPosDfCsv

dfCsv()

# Convert to numpy array
xLevel = salPosDf.iloc[:,0:1].values

X = np.unique(xLevel, return_inverse=True)[1].tolist()
X = pd.DataFrame(X, columns = ["level"])

y = salPosDf.iloc[:,-1:]
# print(np.unique(X))

### Linear regression ###
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

y_predict = lin_reg.predict(X)

## Linear Regression Visualization ##
linearFig = plt.figure(figsize = (10,7))
plt.plot(X, y_predict, color = 'g')
plt.scatter(X, y_predict, color = 'r')
plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.savefig("linearFig.png", dpi=300)

# Discard null TRY salaries
df = df[df.salary_for_tl_currency.isnull() != True]
# Drop not used columns: other currenies column, TRY salary, company size, work_type, currency
df = df.drop("salary_for_other_currency", axis = 1)
df = df.drop("salary_for_tl_currency", axis = 1)
df = df.drop("company_size", axis = 1)
df = df.drop("work_type", axis = 1)
df = df.drop("currency", axis = 1)
df = df.drop("tech_stack", axis = 1)
# Add new min TRY salaries
df["salaries"] = y

# Tranform experience level to numeric values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df["experience"] = le.fit_transform(df["experience"])
df["salaries"] = df["salaries"].astype('float64')

## Count Plot ##
count_plot = sns.countplot(data = df, x = 'salaries', hue = 'level', linewidth = 5) #df["salaries"]
sbnfig = count_plot.get_figure()
sbnfig.savefig("sbnfig.png", dpi = 300)

## Joint Plot ##
joint_plot = sns.jointplot(data = df, x = "salaries", y = "experience", hue = "level")
joint_plot.savefig("snsfig.png", dpi = 300)

## Pairplot ##
pair_plot = sns.pairplot(data = df, hue = "level")
pair_plot.savefig("pairplot.png", dpi = 300)

print("Succesful!")

# print(df.describe())
# print(len(df))
# print(df.isnull().sum())
# df = df.iloc[:,0:-2]
