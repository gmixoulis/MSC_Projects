import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import  silhouette_score

import matplotlib.pyplot as plt


input_data = pd.read_csv("Mall_Customers.csv")
x = input_data.iloc[:,[3,4]].values

# Elbow Method

scorek = []
scoreS =[]
for cluster in range(1,11):
    kmeans = KMeans(n_clusters = cluster,  random_state=10)
    kmeans.fit(x)
    scorek.append(kmeans.inertia_)

    #spectral = SpectralClustering(n_clusters = cluster,  random_state=10)
    #spectral.fit(x)
    #scoreS.append(spectral.inertia_)

# plotting the score

plt.plot(range(1,11), scorek)
plt.title('The Elbow Method for Kmeans')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()




for n_clusters in range(2,11):
    

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(x)

    clustererS = SpectralClustering(n_clusters=n_clusters, random_state=10)
    Sp_cluster_labels = clustererS.fit_predict(x)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(x, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The KMEANS average silhouette_score is :", silhouette_avg, "\n", file=open("output.txt", "a"))
    

    silhouette_avg1 = silhouette_score(x, Sp_cluster_labels)
    print("For n_clusters =", n_clusters,
          "The SPECTRAL average silhouette_score is :" , silhouette_avg1, "\n", file=open("output.txt", "a"))
