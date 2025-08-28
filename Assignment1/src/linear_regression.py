import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

class LinearRegression():

    def __init__(self, learning_rate=0.0001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None   # weight
        self.b = 0      # bias

    def fit(self, X, y):
        """
        Estimates parameters (w, b) for the classifier.
        Uses the following cost equation:

            J(m,b) = 1/n * sum(y - (w*X + b))^2 for i < n

        Args: 
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """

        n_samples, n_features = X.shape

        # initialize params
        self.w = np.zeros(n_features)
        self.b = 0

        # gradient descent loop
        for _ in range(self.epochs):
            # Calculate predictions
            y_pred = X @ self.w + self.b 

            dJ_dw = (-2./n_samples) * X.T @ (y - y_pred) # differentiate wrt w
            dJ_db = (-2./n_samples) * np.sum(y - y_pred) # differenitate wrt b

            # update params
            self.w -= self.learning_rate * dJ_dw
            self.b -= self.learning_rate * dJ_db

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args: 
            X (array<m,n>): a matrix of floats with
            m rows (#samples) and n columns (#features)

        Returns: 
            A length m array of floats
        """
        return X @ self.w + self.b
   
    def plot_prediction_error_dist(self, X, y):
        """
        Plots the Probability Density Function (PDF) of the prediction error dist.

        Should be called after .fit()
        """
        y_pred = self.predict(X)

        errors = (y - y_pred).flatten()

        # Calculate PDF using Gaussian kernel density estimation
        kde = gaussian_kde(errors, bw_method=0.5)
        error_range = np.linspace(min(errors), max(errors), 100)
        pdf = kde(error_range)

        plt.plot(error_range, pdf, color='red')
        plt.title("PDF of Prediction Erorrs")
        plt.xlabel("Error")
        plt.ylabel("Density")
        plt.show()



        
