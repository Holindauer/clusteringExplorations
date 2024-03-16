import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from affinityPropogation import AffinityPropogation 
from spectralClustering import SpectralClustering


def test_affinity_propagation_on_blobs():
    """
    test_affinity_propagation_on_blobs() generates a toy dataset (blobs) with more globular
    characteristics and runs the affinity propogation algorithm on the dataset. The results are 
    then plotted.
    """

    # Generate a toy dataset (blobs) with more globular characteristics
    X, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=42)
    
    # Initialize AffinityPropagation with potentially suitable self-similarity
    # Adjust selfSimilarity if necessary based on the dataset characteristics
    ap = AffinityPropogation(maxIter=500, maxConvergenceIters=15, selfSimilarity=-50)
    
    # Run clustering
    cluster_assignments = ap.cluster(X)
    
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.title('Affinity Propagation on Blobs Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def test_spectral_clustering_on_blobs():
    """
    test_spectral_clustering_on_blobs() generates a toy dataset (blobs) with clear cluster structure
    and runs the spectral clustering algorithm on the dataset. The results are then plotted.
    """
    # Generate a toy dataset (blobs) with globular characteristics
    X, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=42)
    
    # Initialize SpectralClustering with appropriate sigma and number of clusters
    sc = SpectralClustering(numClusters=5, sigma=1.0)
    
    # Run spectral clustering
    labels = sc.fit(X)
    
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.title('Spectral Clustering on Blobs Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


if __name__ == "__main__":
    test_affinity_propagation_on_blobs()
    test_spectral_clustering_on_blobs()
