# Import modules (e.g., numpy), classes (LinearRegression) and functions (r2_score)
import numpy as np
import matplotlib.pyplot as plt


# Custom class that stores the state (parameters) and exposes methods (fit, predict)
class LinearRegression:
    # Constructor method, defines initialisation (internal variables (parameters))
    def __init__(self):
        self.coef_ = np.zeros(1)  # slope placeholder (beta_1)
        self.intercept_ = 0.0  # intercept (bias) placeholder (beta_0)

    # Training (fitting) method, takes X, y and computes beta_0 and beta_1 using OLS (Ordinary Least Squares)
    def fit(self, X, y):
        # OLS assumes for one sample: y_pred_i = beta_0 + beta_1 * x_i
        # To express this for all samples, we add a column of ones to X to form
        # a design matrix which we call X_b [1  X], giving the matrix equation: y_pred = [1  X] [beta_0  beta_1]^T
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))

        # Compute OLS parameters using the Moore–Penrose inverse: beta = (X_b.T X_b)^+ X_b.T y
        # compactly computed as pinv(X_b) @ y
        beta = np.linalg.pinv(X_b) @ y

        # Extract model parameters
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

    # Compute predictions using the learned parameters i.e., y_pred = intercept_ + coef_ * X
    def predict(self, X):
        return self.intercept_ + self.coef_ * X


# Function to compute coefficient of determination (R^2)
# R^2 = 1 - SSR / SST
# where
# SSR (sum of squared residuals) = n * MSE(model) = total squared prediction error of the model (before dividing by n)
# SST (total sum of squares)     = n * MSE(mean)  = total squared difference between each y and the mean of y
# I.e., R^2 measures how much better the model is compared to just predicting the average of y.
def r2_score(y_true, y_pred):
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ssr / sst


# Update runtime configuration dictionary to increase DPI (dots-per-inch) for figures
plt.rcParams["figure.dpi"] = 200

# Generate a synthetic reproducible dataset using seed
# X (also called a feature matrix) from a uniform distribution in range [0, 10)
# y including random noise from a standard normal distribution and an offset
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.squeeze() + np.random.randn(100) * 2 + 5

# Create a linear regression (ordinary least squares) model and fit (train) model on X and y
# After fitting, learned parameters (model.coef_, model.intercept_) are stored in RAM
# and used by model.predict to compute y_pred
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)  # coef. of determination

# Sort X and corresponding predictions for a smooth line plot (avoids zigzagging of ax.plot)
idx = np.argsort(X.squeeze())
X_sort, y_pred_sort = X[idx], y_pred[idx]

# Create a scatter plot of the data and overlay the fitted regression line.
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y, color="steelblue", alpha=0.7, label="Data")
ax.plot(
    X_sort, y_pred_sort, color="darkorange", linewidth=2, label=f"Fit (R² = {r2:.3f})"
)
ax.set_title("Linear Regression")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
plt.tight_layout()
plt.show()
