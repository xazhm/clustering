import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = np.array([
    [10, 20, 30, 40],
    [40, 30, 20, 10],
    [11, 12, 13, 14],
    [20, 19, 18, 10],
    [80, 90, 10, 15],
    [12, 13, 14, 15],
    [35, 45, 55, 65],
    [90, 80, 70, 60],
    [25, 15, 55, 50],
    [10, 10, 10, 10],
    [15, 11, 30, 30],
    [18, 12, 35, 10],
    [23, 15, 12, 5],
    [10, 18, 45, 45],
    [20, 18, 40, 50],
    [30, 20, 15, 25],
    [10, 25, 25, 40],
    [10, 25, 35, 20],
    [80, 25, 30, 30],
    [25, 30, 15, 10],
    [30, 15, 10, 50],
    [33, 50, 40, 30],
    [20, 15, 10, 50],
    [70, 60, 40, 20],
    [50, 35, 45, 50]
])

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Membuat model KMeans dengan 5 klaster 
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42) 
kmeans.fit(data_scaled)  

print("Centroid dari setiap klaster:")
print(scaler.inverse_transform(kmeans.cluster_centers_))  

print("\nKlaster yang ditugaskan untuk setiap data:")
print(kmeans.labels_)