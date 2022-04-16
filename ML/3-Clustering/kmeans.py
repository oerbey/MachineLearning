import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('customers.csv')

# print(df.describe())

X = df.iloc[:,2:4].values

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

# Show clusters
# print(kmeans.cluster_centers_)

results = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    results.append(kmeans.inertia_)

print(results)

kmeansfig = plt.figure(figsize = (11,7))
plt.plot(range(1,11), results)
plt.savefig("kmeansfig.png", dpi=300)
