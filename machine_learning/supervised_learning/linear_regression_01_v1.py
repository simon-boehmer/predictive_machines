# Import modules (e.g., numpy), classes (LinearRegression) and functions (r2_score)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Update runtime configuration dictionary to increase DPI (dots-per-inch) for figures
plt.rcParams["figure.dpi"] = 200

# Generate a synthetic reproducible dataset
# X from a continuous uniform distribution in range [0, 10)
# y including random noise from a standard normal distribution and an offset
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # (100, n) where n = 100
y = 2.5 * X.squeeze() + np.random.randn(100) * 2 + 5  # (100,)

# Create a linear regression model (model = LinearRegression()) and fit (train) model on X and y
# After fitting, learned parameters (model.coef_, model.intercept_) are stored in memory
# and used by model.predict to compute y_pred
model = LinearRegression()  # using ordinary least squares (OLS)
model.fit(X, y)
y_pred = model.predict(X)
R2 = r2_score(y, y_pred)  # coef. of determination

# Sort X and corresponding predictions for a smooth line plot (avoids zigzagging of ax.plot)
idx = np.argsort(X.squeeze())
X_sort, y_pred_sort = X[idx], y_pred[idx]

# Create a scatter plot of the data and overlay the fitted regression line.
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y, color="steelblue", alpha=0.7, label="Data")
ax.plot(
    X_sort, y_pred_sort, color="darkorange", linewidth=2, label=f"Fit (R² = {R2:.3f})"
)
ax.set_title("Linear Regression")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
plt.tight_layout()
plt.show()
