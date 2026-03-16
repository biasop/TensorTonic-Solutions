import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    w = np.zeros(X.shape[1])
    b = 0.0
    N = X.shape[0]
    for step in range(steps):
        z = w.T @ X.T + b
        w = w - lr * 1/N * X.T @ (_sigmoid(z) - y)
        b = b - lr * 1/N * np.sum((_sigmoid(z) - y))

    return (w, b)