import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load MNIST images and labels from CSV file
def load_mnist(csv_path):
    data_df = pd.read_csv(csv_path, header=None, skiprows=1)
    labels = data_df.iloc[:, 0].values.astype(np.uint8)
    images = data_df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
    return images, labels

# Set the file path for the MNIST CSV file
csv_path = '../../datasets/mnist/mnist_train.csv'

# Load the MNIST dataset
train_images, train_labels = load_mnist(csv_path)

# Render the first 30 images
fig, axes = plt.subplots(5, 6, figsize=(10, 8))
axes = axes.flatten()

for i in range(30):
    axes[i].imshow(train_images[i], cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'Label: {train_labels[i]}')

plt.tight_layout()
plt.show()
