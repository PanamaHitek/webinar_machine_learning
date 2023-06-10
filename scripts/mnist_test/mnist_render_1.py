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

# Define the index of the image to render
image_index = 10

# Check if the chosen index is within the valid range
if image_index >= 0 and image_index < len(train_images):
    # Render the chosen image
    plt.imshow(train_images[image_index], cmap='gray')
    plt.title(f'Label: {train_labels[image_index]}')
    plt.axis('off')
    plt.show()
else:
    print("Invalid image index. Please choose an index within the valid range.")
