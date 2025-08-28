import numpy as np

from linear_regression import LinearRegression

def accuracy(y, y_pred):
    return np.mean(y == y_pred)

class LogisticRegression(LinearRegression):

    def __init__(self, learning_rate=0.1, epochs=1000):
        super().__init__(learning_rate, epochs)
        self.losses = []
        self.train_accuracies = []

    @staticmethod
    def _sigmoid(x):
        """
        Sigmoid function 
        """
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1+np.exp(x)))

    def _compute_gradients(self, X, y, y_pred):
        """
        Helper function to compute the two gradients from the loss equation
        """

        # derivative of the loss function
        diff = y_pred - y
        grad_b = np.mean(diff)
        grad_w = X.T @ diff
        grad_w = np.array([np.mean(grad) for grad in grad_w])

        return grad_w, grad_b

    def _compute_loss(self, y, y_pred):
        """
        Helper function to compute the binary cross entropy
        """

        y_zero_loss = y * np.log(y_pred + 1e-9)
        y_one_loss = (1-y) * np.log(1 - y_pred + 1e-9)
        return - np.mean(y_zero_loss + y_one_loss)


    def fit(self, X, y):
        """
        Estimates parameters weight and bias (w,b) for the classifier. 
        Args: 
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats

        """
        _ , n_features = X.shape

        # initalize variables
        self.w = np.zeros(n_features)
        self.b = 0

        # gradient descent loop
        for _ in range(self.epochs):
            x_weights = X @ self.w + self.b
            # y_pred = np.array([self._sigmoid(val) for val in x_weights])
            y_pred = self._sigmoid(x_weights)

            dw, db = self._compute_gradients(X, y, y_pred)
            loss = self._compute_loss(y, y_pred)
            pred_to_class = (y_pred >= 0.5).astype(int)

            self.train_accuracies.append(accuracy(y, pred_to_class))
            self.losses.append(loss)

            # update params
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db


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
        x_weights =  X @ self.w + self.b
        y_pred = self._sigmoid(x_weights)
        return (y_pred >= 0.5).astype(int)


