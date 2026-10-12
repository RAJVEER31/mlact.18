# Applications & Challenges of Unsupervised ML Algorithms
### Activity: Customer Segmentation & Anomaly Detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

database = {
    "CustomerID": range(1, 31),
    "AI": [15, 16, 17, 18, 19, 20, 45, 46, 47, 48, 49, 50, 80, 81, 82, 83, 84, 85, 150, 5, 35, 62, 70, 90, 100, 110, 120, 130, 140, 60],
    "SS": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72, 14, 99, 15, 77, 13, 79, 35, 66, 10, 95, 24, 75, 5, 73, 14, 82, 32, 61, 31, 55],
}

df = pd.DataFrame(database)
df.head()

plt.scatter(df["AI"], df["SS"])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Raw Customer Data")
plt.show()

X = df[["AI", "SS"]]
kmeans = KMeans(n_clusters=3, random_state=42)
df["Segment"] = kmeans.fit_predict(X)

plt.scatter(df["AI"], df["SS"], c=df["Segment"], cmap="plasma")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c="red", marker="X", s=200)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.show()

centroids = kmeans.cluster_centers_
df["DTS"] = df.apply(
    lambda row: np.linalg.norm([row["AI"], row["SS"]] - centroids[row["Segment"]]),
    axis=1
)
threshold = df["DTS"].mean() + 2 * df["DTS"].std()
df["Anomaly"] = df["DTS"] > threshold
df[df["Anomaly"] == True]

plt.scatter(df["AI"], df["SS"], c=df["Segment"], cmap="plasma")
plt.scatter(df[df["Anomaly"]]["AI"], df[df["Anomaly"]]["SS"], c="red", marker="*", s=300, label="Anomaly")
plt.legend()
plt.title("Detected Anomalies")
plt.show()

inertia = []
k_range = range(1, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(k_range, inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()