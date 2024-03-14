import numpy as np
from numpy import ndarray

# affinityPropogation.py

class AffinityPropogation:
    """
    The affinity propogation algorithm is an exemplar based clustering algorith. It involves
    three square matrices of size (numExamples, numExamples): affinity, responsibility, and 
    availability. 

        - The affinity matrix contains the similarity (negative euclian distance) between 
          the i'th and k'th example in the dataset.

        - At each (i, j) element of the responsibility matrix is a reflection on how well the 
          k'th example is suited to be the exemplar for the i'th example.

        - At each (i, j) element of the availability matrix is a refleciton on how well the 
          k'th example is suited to be the exemplar of the i'th example when considering the 
          support from other data points in the responsibility matrix.
    """

    def __init__(self, maxIter=200, maxConvergenceIters=10, selfSimilarity=-50):

        # maxIter is the maximum number of iterations to run the algorithm
        self.maxIter = maxIter

        # maxConvergenceIters is the number of iterations to wait for convergence
        # once the exemplar assignments have not changed
        self.maxConvergenceIters = maxConvergenceIters

        # selfSimilarity is the similarity of each example to itself.
        self.selfSimilarity = selfSimilarity

        self.numExamples = 0

    def initMatricies(self) -> None:
        """ initMatricies initializes the three square matrices used within
        the affinity propogation algorithm."""     

        # similarity matrix
        self.affinity = np.zeros((self.numExamples, self.numExamples))
        if self.selfSimilarity is not None:
            np.fill_diagonal(self.affinity, self.selfSimilarity)

        # responsibility matrix
        self.responsibility = np.zeros((self.numExamples, self.numExamples))

        # availability matrix
        self.availability = np.zeros((self.numExamples, self.numExamples))

    def updateResponsibilities(self) -> None:
        """ The responsibility matrix is updated by setting each element (i, k) to the difference 
        between the affinity(i, k) and the maximum of the sum of the affinity and availability
        excluding the diagonal."""

        # affintiy and availability mat sum
        AS: ndarray = (self.affinity + self.availability).copy()
        np.fill_diagonal(AS, -np.inf) # exclude diagonal in .max() by setting to -inf

        # apply responsibility matrix update to class variable 
        # NOTE Keepdims ensures shape is (N,1) for correct broadcasting
        self.responsibility = self.affinity - np.max(AS, axis=1, keepdims=True)  

    def updateAvailabilities(self) -> None:
        """ 
        The availability matrix is updated differently depending on if it is a self availability 
        (diagonal element) or not. 

        For non-self availabilities:
            - The update sets each element (i, k) to the minimum of 0 or the self responsibility
              responsibility(k, k) plus the sum of the positive elements of the responsibility
              matrix excluding the diagonal.

        For self availabilities:
            - The update sets each element (k, k) to the sum of the positive elements of the 
              responsibility matrix excluding the diagonal.
        """
        
        # copy the responsibility matrix to manipulate
        resCopy = self.responsibility.copy()
        np.fill_diagonal(resCopy, 0)  # Set diagonal elements to 0 
        
        # get positive responsibilities and sum of positive responsibilities
        posRes = np.maximum(resCopy, 0)
        sum_posRes = np.sum(posRes, axis=0) - posRes
        
        # update non-self availabilities
        self.availability = np.minimum(0, sum_posRes + np.diag(self.responsibility))
        
        # update self availabilities
        np.fill_diagonal(self.availability, np.sum(posRes, axis=0))

    def computeSimilarityMat(self, dataset):
        """Compute the similarity matrix based on negative Euclidean distance using vectorized operations."""

        # broadcast dataset(:, 1, :) - dataset(1, :, :) and square result
        # This calculates the squared differences between each pair of examples
        diff_sq = np.sum(
            (dataset[:, np.newaxis, :] - dataset[np.newaxis, :, :]) ** 2, 
            axis=-1
            )
        
        # Assign affinity matrix the negative root of the squared differences to get negative distances
        self.affinity = -np.sqrt(diff_sq)
        
        # Set the diagonal to self.selfSimilarity
        if self.selfSimilarity is not None:
            np.fill_diagonal(self.affinity, self.selfSimilarity)
    
    def check_convergence(self, old_exemplars, new_exemplars): 
        """Are exemplar assignments the same as the previous iteration?"""
        return np.array_equal(old_exemplars, new_exemplars) 

    def identifyExemplars(self):
        """Identify exemplars based on the sum of responsibility and availability."""
        return np.argmax((self.responsibility + self.availability), axis=1)

    def cluster(self, dataset: ndarray) -> ndarray:
        """.cluster applies the full affinity propogation algorithm to the dataset  
        and returns the cluster assignments for each example in the dataset."""

        # setup matrices and compute similarity matrix
        self.numExamples = dataset.shape[0]
        self.initMatricies()
        self.computeSimilarityMat(dataset)
        
        # init vars for convergence check
        numConvergedIters = 0
        old_exemplars = np.zeros(self.numExamples, dtype=int)

        # Run the algorithm for maxIter iterations
        for iteration in range(self.maxIter):

            # apply update rules
            self.updateResponsibilities()
            self.updateAvailabilities()

            # find exemplars
            newExemplars = self.identifyExemplars()
            
            # check convergence
            if self.check_convergence(old_exemplars, newExemplars):
                numConvergedIters += 1
                if numConvergedIters >= self.maxConvergenceIters:
                    print(f"Converged after {iteration + 1} iterations.")
                    break
            else:
                numConvergedIters = 0

            old_exemplars = newExemplars.copy()

        # return final exemplar/cluster assignments
        return newExemplars