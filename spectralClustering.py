from sklearn.cluster import KMeans
import numpy as np
from numpy import ndarray

class SpectralClustering:
    """
    The spectral clusteirng algorithm uses the eigenvalues of the similarity matrix to reduce
    the dimmensionality of a dataset before applying more traditional clustering algorithms like 
    kmeans to the datset.
    """

    def __init__(self, numClusters: int, sigma: float = 1.0,):

        # neighborhood width of the Gaussian kernel
        self.sigma = sigma               

        # init similarity, laplacian, and selected eigenvectors to None
        self.similarity = None           
        self.laplacian = None          
        self.selectedEigenvectors = None 

        # instantiate kmeans clustering
        self.numClusters = numClusters
        self.kmeans = KMeans(n_clusters=numClusters, random_state=0)

    def similarityMat(self, dataset : ndarray) -> None:
        """
        similarityMat() computes the similarity matrix of the dataset using the Gaussian kernel.
        The Gaussian kernel is defined as: exp(-||x_i - x_j||^2 / 2 * sigma^2). Where sigma is a
        hyperparameter that controls the neighborhood width of the Gaussian kernel. This is done 
        using vectorized operations to avoid using for loops. The similarity matrix is then returned
        """


        # broadcast dataset[:, 1, :] - dataset[1, :, :] across dataset, square results, and sum along  last axis
        # this calculates the squared Euclidean distance matrix in a vectorized manner
        euclidDistMat = np.sum( (dataset[:, np.newaxis, :] - dataset[np.newaxis, :, :]) ** 2,  axis=-1)

        # apply the Gaussian kernel to the squared differences
        assert self.sigma is not None
        self.similarity = np.exp(-euclidDistMat / (2 * self.sigma**2))
    
    def laplacianMat(self) -> None:
        """ 
        wrt graph theory, a laplacian matrix represents the connectivity of graph vertices such that differences 
        in connectivity are highlighted. A graph with n vertices has the graph Laplacian defined by L = D - A.

        Where:
            A is the adjacency matrix. A_ij is non zero (typically 1) if there is an edge from vetex i to j, else 0. 
            D is the degree matrix which is a diagonal matrix containing the degree of vetrex i. .

        In spectral clustering a similarity graph (matrix) is constructed with each example being a node. The edges 
        between nodes are weighted by the Gaussian kernel. L = D-A is computed with A as the similarity matrix and 
        D as the diagonal matrix computed by summing the similarities across the j axis for each example i.
        """
        assert self.similarity is not None, "Similarity matrix must be computed before the Laplacian."
        
        # Degree matrix: diagonal matrix of node degrees
        degree_matrix = np.diag(np.sum(self.similarity, axis=1))
        
        # Unnormalized graph Laplacian
        self.laplacian = degree_matrix - self.similarity


    def eigenValueDecomp(self) -> None:
        """
        In linear algebra, for a square matrix A, an eigenvector is a non-zero vector that only changes by a scaling 
        factor when the linear transformation A is applied to it. The scaling factor is the eigenvalue for a given 
        eigenvector. 

        wrt spectral clustering, an eigenvalue decomposition transforms the lapacian graph of the dataset into a smaller 
        dimensional space. The features then are the smallest non-trivial eigenvalues. This compresses the dataset into 
        a smaller space.
        """
        assert self.laplacian is not None, "Laplacian matrix must be computed before eigenvalue decomposition."
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.laplacian)

        # Ensure there's at least one non-zero eigenvalue to skip
        assert np.count_nonzero(eigenvalues) > self.numClusters, "Not enough non-zero eigenvalues for the desired number of clusters."

        # Select eigenvectors corresponding to the 'numClusters' smallest non-zero eigenvalues
        idx = np.argsort(eigenvalues)[1:self.numClusters+1]  # Skip the first eigenvalue if it's zero
        self.selectedEigenvectors = eigenvectors[:, idx]


    def kMeans(self):
        """
        Once the eigenvalue decomposition is complete on the laplacian matrix, the selected eigenvectors are used as
        the features for the dataset. The dataset is then clustered using kmeans. 
        """ 
        assert(self.selectedEigenvectors is not None), "Eigenvalue decomposition must be computed before kmeans."

        # apply kmeans to the selected eigenvectors
        self.kmeans.fit(self.selectedEigenvectors)

    def assignLabels(self):
        """
        After applying kmeans to the selected eigenvectors, the labels are assigned to the dataset.
        """
        assert self.kmeans.labels_ is not None, "Kmeans must be computed before assigning labels."

        # assign labels to the dataset
        self.labels = self.kmeans.labels_

    def fit(self, dataset: ndarray) -> ndarray:
        """
        fit() is the main method for the spectral clustering algorithm. It computes the similarity matrix,
        laplacian matrix, eigenvalue decomposition, kmeans, and assigns labels to the dataset.
        """

        # reset similarity, laplacian, and selected eigenvectors
        self.similarity = None
        self.laplacian = None
        self.selectedEigenvectors = None
        self.labels = None

        # fasciliate entire spectral clustering process
        self.similarityMat(dataset)
        self.laplacianMat()
        self.eigenValueDecomp()
        self.kMeans()
        self.assignLabels()

        return self.labels

        
