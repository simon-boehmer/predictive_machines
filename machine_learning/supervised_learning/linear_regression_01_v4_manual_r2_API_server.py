# Import modules (e.g., numpy)
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)


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
