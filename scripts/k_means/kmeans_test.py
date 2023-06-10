import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load MNIST images and labels from CSV file
def load_mnist(csv_path):
    data_df = pd.read_csv(csv_path, header=None, skiprows=1)
    labels = data_df.iloc[:, 0].values.astype(np.uint8)
    images = data_df.iloc[:, 1:].values.astype(np.uint8)
    return images, labels

# Set the file path for the MNIST CSV file
csv_path = '../../datasets/mnist/mnist_train.csv'

# Load the MNIST dataset
images, labels = load_mnist(csv_path)

# Reshape the images to be 2D (flatten)
images_flat = images.reshape(images.shape[0], -1)

# Perform k-means clustering with 10 clusters
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(images_flat)

# Get the cluster labels for each image
cluster_labels = kmeans.labels_

# Create a dictionary to store the indices of the first 10 samples of each cluster
cluster_samples = {i: [] for i in range(10)}

# Find the indices of the first 10 samples of each cluster
for i, label in enumerate(cluster_labels):
    if len(cluster_samples[label]) < 10:
        cluster_samples[label].append(i)

# Plot the first 10 samples of each cluster with a single label
fig, axes = plt.subplots(10, 11, figsize=(12, 10))
for i in range(10):
    axes[i, 0].text(0.5, 0.5, f'Cluster: {i}', fontsize=12, ha='center', va='center')
    axes[i, 0].axis('off')
    for j in range(10):
        index = cluster_samples[i][j]
        image = images[index].reshape(28, 28)  # Reshape the image back to 2D
        axes[i, j+1].imshow(image, cmap='gray')
        axes[i, j+1].axis('off')

plt.tight_layout()
plt.show()

# Print the count of samples in each cluster
for i in range(10):
    count = len(cluster_samples[i])
    print(f"Cluster {i}: {count} samples")
