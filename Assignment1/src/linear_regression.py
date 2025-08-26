import numpy as np

class LinearRegression():

    def __init__(self, learning_rate=0.0001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None   # weight
        self.b = 0      # bias

    def fit(self, X, y):
        """
        Estimates parameters (w, b) for the classifier.
        Uses the following equation (MSE)

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
        return X @ self.w + self.b

