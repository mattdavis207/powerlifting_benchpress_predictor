import pandas as pd
import numpy as np 
import joblib 
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error


X_train = pd.read_csv("preprocessed_data/X_train.csv")
X_test = pd.read_csv("preprocessed_data/X_test.csv")
y_train = pd.read_csv("preprocessed_data/y_train.csv")
y_test = pd.read_csv("preprocessed_data/y_test.csv")


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

print(root_mean_squared_error(y_test, y_pred))

# cross_val_scores = cross_val_score(lin_reg, X_train, y_train, scoring = "accuracy", )
# print(cross_val_scores)



