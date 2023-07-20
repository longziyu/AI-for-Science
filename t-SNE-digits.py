import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

# Plot the resulting 2D scatter plot
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.colorbar()
plt.show()
