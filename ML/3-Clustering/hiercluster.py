# https://scikit-learn.org/stable/modules/clustering.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('customers.csv')

# print(df.describe())

X = df.iloc[:,2:4].values

## Hierarchial Clustering
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = 'ward')

Y_predict = ac.fit_predict(X)
print(Y_predict)

plt.scatter(X[Y_predict == 0, 0], X[Y_predict == 0, 1], s = 100, c = 'red')
plt.scatter(X[Y_predict == 1, 0], X[Y_predict == 1, 1], s = 100, c = 'blue')
plt.scatter(X[Y_predict == 2, 0], X[Y_predict == 2, 1], s = 100, c = 'green')
plt.title('Hierarchial Cluster')
hierplt = plt.show()
plt.savefig("hierplt.png", dpi=300)

## Dendogram
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
dendplt = plt.show()
plt.savefig("dendplt.png", dpi = 300)
