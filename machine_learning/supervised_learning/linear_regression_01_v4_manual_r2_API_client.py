# Import modules (e.g., numpy), classes (LinearRegression) and functions (r2_score)
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression


# Update runtime configuration dictionary to increase DPI (dots-per-inch) for figures
plt.rcParams["figure.dpi"] = 200

# Generate a synthetic reproducible dataset using seed
# X (also called a feature matrix) from a uniform distribution in range [0, 10)
# y including random noise from a standard normal distribution and an offset
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # (100, n) where n = 100
y = 2.5 * X.squeeze() + np.random.randn(100) * 2 + 5  # (100,) 1D

# Create a linear regression model (model = LinearRegression()) and fit (train) model on X and y
# After fitting, learned parameters (model.coef_, model.intercept_) are stored in memory
# and used by model.predict to compute y_pred
model = LinearRegression()  # using ordinary least squares (OLS)
model.fit(X, y)
y_pred = model.predict(X)

# API for r2 score computation
# BASE_URL defines the root address of Flask API (localhost 127.0.0.1 on TCP port 8000)
# -> URL for the r2 score API endpoint is http://127.0.0.1:8000/r2 (see ...API_server.py)
# FYI:
# Even though both processes (client / server) run locally, TCP serves as the communication channel between them.
# We could use other forms of inter-process communication (e.g., direct function calls), but since Flask implements
# HTTP (a web protocol), it inherently relies on TCP (the full network stack as defined in the OSI model).
BASE_URL = "http://127.0.0.1:8000"
# HTTP POST request to API endpoint /r2 using Pythons "request" library for HTTP calls
# FYI:
# We need .tolist() to convert NumPy arrays (ndarray) into standard Python lists.
# This conversion makes the data JSON-serializable for HTTP transmission (JSON cannot encode NumPy objects directly).
resp = requests.post(
    f"{BASE_URL}/r2", json={"y_true": y.tolist(), "y_pred": y_pred.tolist()}
)
# resp.json() parses (i.e., transforms a string into an actual Python data structure)
# from JSON into e.g., a dictionary {"r2": 0.95}
r2 = resp.json()["r2"]

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
