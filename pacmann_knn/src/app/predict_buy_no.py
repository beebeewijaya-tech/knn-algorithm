from pacmann_knn.src.models.KNN import KNN
from pacmann_knn.src.utils.plot import plotting_predict_buy_no
import pandas as pd
import numpy as np

buy_no_file = pd.read_csv("../../data/cleaned/buying_or_no.csv")
X = buy_no_file[["EstimatedSalary"]].values
y = buy_no_file["Purchased"].values


K_value = []
y_predict_value = []
for i in range(1, 101):
    # Declare input for predict test
    X_input = 33000

    # Instantiate KNN
    knn = KNN(k_neighbors=i)

    # Fit the model
    knn.fit(X, y)

    # Predict the X_input
    y_pred = knn.predict(X_input)

    # Append the k-value
    K_value.append(i)

    # Append the predicted value
    y_predict_value.append(y_pred)

# Plot the k-value <> prediction
plotting_predict_buy_no(np.array(K_value), np.array(y_predict_value))
