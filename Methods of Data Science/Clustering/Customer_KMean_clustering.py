# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns 
from sklearn.metrics import silhouette_score

df = pd.read_csv(r'C:/Users/86198/IS517/week2 K-means/Mall_Customers.csv')
df.columns
df = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Caculate the Within-Cluster-Sum-of-Squares of various numer of clusters
wcss = []
for i in range(1, 8):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

# Print the plot for selecting the number of clusters
# the optimal K = 2
sns.set()
plt.plot(range(1, 8), wcss)
plt.title('Selecting the Number of Clusters using the Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('Within Cluster Sum of Squares')
plt.show()

kmeans = KMeans(n_clusters=2).fit(df)
print("Cluster Centroid Values:")
centroids = kmeans.cluster_centers_
print(centroids)

print('Labels of each point')
labels = kmeans.labels_
print(labels)

#Caculate the silhouette_score 
label = kmeans.fit_predict(df)
kmeans_silhouette = silhouette_score(df, label)
print("Overall Average Silhouette Score:")
print(kmeans_silhouette)

plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='blue', s=50)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

