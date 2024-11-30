import numpy as np
from scipy.linalg import svd


class SVDModel:
    def __init__(self, dataframe, n_components):
        """
        Initialize the SVD model.

        :param dataframe: Input pandas DataFrame to apply SVD on.
        :param n_components: Number of components to retain after SVD.
        """
        self.dataframe = dataframe
        self.n_components = n_components
        self.u = None
        self.s = None
        self.vt = None
        self.reduced_data = None

    def fit(self):
        """
        Perform Singular Value Decomposition (SVD) on the dataset.
        """
        # Convert the DataFrame to a NumPy array
        data_matrix = self.dataframe.to_numpy()

        # Perform SVD: A = U * S * Vt
        self.u, self.s, self.vt = svd(data_matrix, full_matrices=False)

    def transform(self):
        """
        Reduce the dimensionality of the dataset.

        :return: Reduced data as a numpy array.
        """
        if self.u is None or self.s is None or self.vt is None:
            raise RuntimeError("You need to fit the model before transforming data.")

        # Select the top n_components
        u_reduced = self.u[:, :self.n_components]
        s_reduced = np.diag(self.s[:self.n_components])
        vt_reduced = self.vt[:self.n_components, :]

        # Reduced data
        self.reduced_data = np.dot(u_reduced, s_reduced)
        return self.reduced_data