import numpy as np
import matplotlib.pyplot as plt

class ElasticNetModel:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, learning_rate=0.01):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.costs_ = []

    def _soft_thresholding(self, x, lambda_):
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

    def _calculate_r_squared(self, y, y_pred):
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def _calculate_mse(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def fit(self, X, y):
        m, n = X.shape
        X = np.c_[np.ones((m, 1)), X]
        theta = np.zeros(n + 1)

        for i in range(self.max_iter):
            old_theta = theta.copy()

            for j in range(n + 1):
                if j == 0:  # Intercept
                    theta[j] -= self.learning_rate * np.mean(X.dot(theta) - y)
                else:
                    theta[j] = self._soft_thresholding(
                        theta[j] - self.learning_rate * np.dot(X[:, j], X.dot(theta) - y) / m,
                        self.alpha * self.l1_ratio * self.learning_rate
                    )
                    theta[j] /= (1 + self.learning_rate * self.alpha * (1 - self.l1_ratio))

            cost = self._cost(X, y, theta)
            self.costs_.append(cost)

            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.6f}")

            if np.sum(np.abs(old_theta - theta)) < self.tol:
                print(f"Converged after {i} iterations")
                break

        
        results = ElasticNetModelResults()
        results.coef_ = theta[1:]
        results.intercept_ = theta[0]
        results.costs_ = self.costs_
        
        
        y_pred = results.predict(X[:, 1:])
        results.r_squared_ = self._calculate_r_squared(y, y_pred)
        results.mse_ = self._calculate_mse(y, y_pred)
        
        return results

    def _cost(self, X, y, theta):
        m = len(y)
        h = X.dot(theta)
        cost = (1 / (2 * m)) * np.sum((h - y) ** 2)
        l1_penalty = self.alpha * self.l1_ratio * np.sum(np.abs(theta[1:]))
        l2_penalty = 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(theta[1:] ** 2)
        return cost + l1_penalty + l2_penalty

class ElasticNetModelResults:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.r_squared_ = None
        self.mse_ = None
        self.costs_ = None

    def predict(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X].dot(np.r_[self.intercept_, self.coef_])

    def plot_convergence(self):
        """Plot the convergence of the cost function over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.costs_)
        plt.title('Cost Function over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

    def plot_predictions(self, X, y_true):
        """Plot predicted vs actual values."""
        y_pred = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual Values\nR² = {self.r_squared_:.4f}, MSE = {self.mse_:.4f}')
        plt.grid(True)
        plt.show()

    def plot_feature_importance(self, feature_names=None):
        """Plot feature importance."""
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.coef_))]
        
        plt.figure(figsize=(12, 6))
        feature_importance = np.abs(self.coef_)
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        
        
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.xlabel('Relative Importance (%)')
        plt.title('Feature Importance')
        plt.grid(True)
        plt.show()
