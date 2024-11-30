import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

class GMMCluster:

    def __init__(self, dataframe, k, max_iter=100, tol=1e-4):
        self.k = k
        self.dataframe = dataframe
        self.max_iter = max_iter
        self.tol = tol
        self.n_samples, self.n_features = dataframe.shape

        # Initialize parameters
        self.weights = np.full(k, 1 / k)  # Equal weights
        self.means = dataframe.sample(k).to_numpy()  # Random initial centroids
        self.covariances = [np.cov(dataframe.T) + np.eye(self.n_features) * 1e-6 for _ in range(k)]  # Regularized covariance
        self.responsibilities = np.zeros((self.n_samples, k))  # Responsibility matrix

    def e_step(self):
        for i in range(self.k):
            self.responsibilities[:, i] = self.weights[i] * multivariate_normal.pdf(
                self.dataframe, mean=self.means[i], cov=self.covariances[i], allow_singular=True
            )
        self.responsibilities /= self.responsibilities.sum(axis=1, keepdims=True)

    def m_step(self):
        for i in range(self.k):
            responsibility = self.responsibilities[:, i]
            total_responsibility = responsibility.sum()

            # Update weights
            self.weights[i] = total_responsibility / self.n_samples

            # Update means
            self.means[i] = np.dot(responsibility, self.dataframe) / total_responsibility

            # Update covariances with regularization
            diff = self.dataframe - self.means[i]
            self.covariances[i] = (
                np.dot(responsibility * diff.T, diff) / total_responsibility
                + np.eye(self.n_features) * 1e-6
            )

    def log_likelihood(self):
        likelihood = 0
        for i in range(self.k):
            likelihood += self.weights[i] * multivariate_normal.pdf(
                self.dataframe, mean=self.means[i], cov=self.covariances[i], allow_singular=True
            )
        return np.sum(np.log(likelihood))

    def fit(self):
        prev_likelihood = None
        for iteration in range(self.max_iter):
            self.e_step()
            self.m_step()
            current_likelihood = self.log_likelihood()
            print(f"Iteration {iteration}: Log Likelihood = {current_likelihood}")

            if prev_likelihood is not None and abs(current_likelihood - prev_likelihood) < self.tol:
                break
            prev_likelihood = current_likelihood

    def predict(self):
        return np.argmax(self.responsibilities, axis=1)
