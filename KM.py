import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = "tornados_clear.csv"  # Replace with your dataset path
data = pd.read_csv(file_path)
data = data.sample(frac=0.1)

# Step 2: Select continuous features
continuous_features = ['property_loss', 'start_latitude', 'start_longitude',
                       'end_latitude', 'end_longitude', 'track_length_miles']
X = data[continuous_features]

# Step 3: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Determine the optimal number of clusters (K) using the Elbow Method
inertia = []
silhouette_scores = []
K_range = range(2, 11)  # Test K values from 2 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot the Elbow Method graph
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

# Plot the Silhouette Score graph
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')

plt.tight_layout()
plt.show()

# Step 5: Choose the optimal K based on the Elbow Method and Silhouette Score
optimal_k = int(input("Enter the optimal number of clusters (K) based on the graphs: "))

# Step 6: Perform K-Means clustering with the optimal K
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Add cluster labels to the original dataset
data['Cluster'] = labels

# Step 7: Evaluate clustering quality
inertia = kmeans.inertia_
silhouette_avg = silhouette_score(X_scaled, labels)

print(f"Inertia: {inertia}")
print(f"Silhouette Score: {silhouette_avg}")

# Step 8: Visualize the clusters using PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for visualization
cluster_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
cluster_df['Cluster'] = labels

# Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=cluster_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100)
plt.title('K-Means Clustering (2D PCA Visualization)')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.legend(title='Cluster')
plt.show()
