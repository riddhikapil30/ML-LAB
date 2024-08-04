#Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs.
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.sort(np.random.rand(100))
y = np.sin(2 * np.pi * X) + np.random.randn(100) * 0.1

def locally_weighted_regression(X, y, tau, x_query):
    m = len(X)
    weights = np.exp(-((X - x_query) ** 2) / (2 * tau ** 2))
    W = np.diag(weights)

    X_ = np.vstack([np.ones(m), X]).T
    theta = np.linalg.pinv(X_.T @ W @ X_) @ X_.T @ W @ y

    x_query_ = np.array([1, x_query])
    y_query = x_query_ @ theta

    return y_query

def predict(X, y, tau, x_values):
    y_preds = np.array([locally_weighted_regression(X, y, tau, x) for x in x_values])
    return y_preds

tau = 0.1

x_values = np.linspace(0, 1, 1000)
y_preds = predict(X, y, tau, x_values)

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(x_values, y_preds, color='red', label='LWR Curve')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.show()
