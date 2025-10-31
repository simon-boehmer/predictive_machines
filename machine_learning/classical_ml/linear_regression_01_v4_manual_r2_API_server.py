# Import modules (e.g., numpy)
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)


# Function to compute coefficient of determination (R^2)
# R^2 = 1 - (model error / total variation) where:
# model error (SSR): Sum of squared differences between actual (y_true) and predicted (y_pred) values
# total variation (SST): Sum of squared differences between actual values and their mean
# Intuition: R^2 (0 to 1) shows how much better the model predicts the actual data compared to using just the mean
def r2_score(y_true, y_pred_arg):
    ssr = np.sum((y_true - y_pred_arg) ** 2)  # model error (sum of squared residuals)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)  # total variation in y_true
    return 1 - ssr / sst


# Need to add comments...
@app.post("/r2")
def r2():
    data = request.get_json()
    y_true = np.asarray(data["y_true"], dtype=float)
    y_pred = np.asarray(data["y_pred"], dtype=float)
    if y_true.shape != y_pred.shape:
        return jsonify(error="y_true and y_pred must have the same shape."), 400
    return jsonify(r2=float(r2_score(y_true, y_pred)))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
