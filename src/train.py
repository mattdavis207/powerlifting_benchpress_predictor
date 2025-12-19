import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import matplotlib.pyplot as plt
from scipy.stats import randint 
from xgboost import XGBRegressor


X_train = pd.read_csv("preprocessed_data/X_train.csv")
X_test = pd.read_csv("preprocessed_data/X_test.csv")
y_train = pd.read_csv("preprocessed_data/y_train.csv")
y_test = pd.read_csv("preprocessed_data/y_test.csv")



# Next, let's try different models

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

cross_val_scores = cross_val_score(lin_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=5)

print("\nLinear Regression Results")
print(f"Cross-validation MSE scores: {-cross_val_scores}")  # Negative because sklearn returns negative MSE
print(f"Average MSE: {-cross_val_scores.mean():.2f}")
print(f"Average RMSE: {np.sqrt(-cross_val_scores.mean()):.2f}")


# Ridge Regression
ridge_reg = RidgeCV(alphas=[6.25, 6.5, 6, 6.15, 6.35], store_cv_results=True, cv=None)
ridge_reg.fit(X_train, y_train)

print(f"\nRidge Regression Results")
print(f"Best alpha: {ridge_reg.alpha_}")
print(f"Training R² score: {ridge_reg.score(X_train, y_train):.4f}")
print(f"Test R² score: {ridge_reg.score(X_test, y_test):.4f}")

# Cross-validation scores
ridge_cv_scores = cross_val_score(ridge_reg, X_test, y_test, scoring="neg_mean_squared_error", cv=5)
print(f"Cross-validation RMSE: {np.sqrt(-ridge_cv_scores.mean()):.2f}")

# Compare to Linear Regression
print(f"\nCompare")
print(f"Linear Regression Test R²: {lin_reg.score(X_test, y_test):.4f}")
print(f"Ridge Regression Test R²: {ridge_reg.score(X_test, y_test):.4f}")


# Decision Tree Regression
dec_tree_reg = DecisionTreeRegressor(max_depth=15, criterion='squared_error')
dec_tree_reg.fit(X_train, y_train)

# param_distribution = dict(
#     criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
#     max_depth = randint(1, 100)
# )

# Next let's find best hyperparameters for Decision Tree
# rscv = RandomizedSearchCV(dec_tree_reg, param_distributions=param_distribution, scoring='neg_mean_squared_error', n_iter=10, cv=5, random_state=42, verbose=2)
# rscv.fit(X_train, y_train.values.ravel())  # Flatten y_train to 1D array

print(f"\nDecision Tree Results")
print(f"Training Score: {dec_tree_reg.score(X_train, y_train)}")
print(f"Test Score: {dec_tree_reg.score(X_test, y_test):.2f}")
print(f"Test RMSE: {np.sqrt(dec_tree_reg.score(X_test, y_test)):.2f}")




# # RandomizedSearchCV for Random Forest
# param_dist = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 15, 20, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# rf_search = RandomizedSearchCV(param_distributions=param_dist, estimator=RandomForestRegressor(random_state=42), n_iter=20, cv=5, scoring='neg_mean_squared_error', random_state=42, verbose=2)
# rf_search.fit(X_train, y_train.values.ravel())

# print(f"Best params: {rf_search.best_params_}")
# print(f"Best R² score: {rf_search.best_score_}")



# Random Forest Regression
print(f"\nRandom Forest Regression")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_split=2 ,min_samples_leaf=4, max_depth=10)
rf_reg.fit(X_train, y_train.values.ravel())

print(f"Training R² score: {rf_reg.score(X_train, y_train):.4f}")
print(f"Test R² score: {rf_reg.score(X_test, y_test):.4f}")

# Cross-validation
rf_cv_scores = cross_val_score(rf_reg, X_train, y_train.values.ravel(),
                                 scoring='neg_mean_squared_error', cv=5)
print(f"Cross-validation RMSE: {np.sqrt(-rf_cv_scores.mean()):.2f}")


# Let's try some Boosting, start with GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=500, n_iter_no_change=10, learning_rate=0.05, random_state=42)
gbrt.fit(X_train, y_train.values.ravel())  # Flatten y_train to avoid warning
print(f"\nGradient Boosted Regression Trees")
print(f"Training R² score: {gbrt.score(X_train, y_train):.4f}")
print(f"Test R² score: {gbrt.score(X_test, y_test):.4f}\n")


# param_dist = {
#     'n_estimators': [100, 200, 300, 500],
#     'max_depth': [3, 5, 7, 10],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0]
# }


# xgbr_search = RandomizedSearchCV(
#     XGBRegressor(random_state=42, n_jobs=-1),
#     param_distributions=param_dist,
#     n_iter=20,
#     cv=5,
#     scoring='neg_mean_squared_error',
#     random_state=42,
#     verbose=1
# )

# xgbr_search.fit(X_train, y_train.values.ravel())

# print(f"Best params: {xgbr_search.best_params_}")
# print(f"Best rest R² score: {xgbr_search.best_score_}")


# Lastly, lets try XGBoost
xgbr = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,  # Fixed typo: was sub_sample
    random_state=42,
    n_jobs=-1
)

xgbr.fit(X_train, y_train.values.ravel())
print("XGBoost Regression Trees")
print(f"XGBoost Train R² score: {xgbr.score(X_train, y_train)}")
print(f"XGBoost Test R² score: {xgbr.score(X_test, y_test)}")



# Final Comparison
print(f"\nFinal Model Comparison")
print(f"Linear Regression Test R²:    {lin_reg.score(X_test, y_test):.4f}")
print(f"Ridge Regression Test R²:     {ridge_reg.score(X_test, y_test):.4f}")
print(f"Decision Tree Test R²:        {dec_tree_reg.score(X_test, y_test):.4f}")
print(f"Random Forest Test R²:        {rf_reg.score(X_test, y_test):.4f}")
print(f"Gradient Boosted Regression Trees R²: {gbrt.score(X_test, y_test):.4f}")
print(f"XGBoost Test R² score: {xgbr.score(X_test, y_test)}")



# y_pred_tree = dec_tree_reg.predict(X_test)

# print(f"\nDecision Tree Results")
# print(f"Training R² score: {dec_tree_reg.score(X_train, y_train):.4f}")
# print(f"Test R² score: {dec_tree_reg.score(X_test, y_test):.4f}")

# Plot for Predicted vs Actual
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred_tree, alpha=0.5, edgecolor="black", c="darkorange", label="Predictions")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Prediction")
# plt.xlabel("Actual Bench Press (kg)")
# plt.ylabel("Predicted Bench Press (kg)")
# plt.title("Decision Tree Regression: Predicted vs Actual")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()











