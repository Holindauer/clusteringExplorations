import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from affinityPropogation import AffinityPropogation  # Make sure this is the corrected class name and file

def test_affinity_propagation_on_blobs():
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

if __name__ == "__main__":
    test_affinity_propagation_on_blobs()
