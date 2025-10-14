import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Update runtime config. dict to increase DPI (dots-per-inch) for figures
plt.rcParams["figure.dpi"] = 200

# Generate synthetic reproducible (using seed) dataset
# X (feature matrix) from a uniform distribution in range [0, 10)
# y incl. random noise from a standard normal distribution + offset
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.squeeze() + np.random.randn(100) * 2 + 5

# Create linear regression (ordinary least squares) model and fit (train) model on X
# model.coef_ and model.intercept_ stored and used to compute y_pred
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)  # coef. of determination

# Sort for line plot (avoid zigzagging of ax.plot)
idx = np.argsort(X.squeeze())
X_sort, y_pred_sort = X[idx], y_pred[idx]

# Plot
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
