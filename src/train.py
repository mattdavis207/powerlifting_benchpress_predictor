import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import matplotlib.pyplot as plt
from scipy.stats import randint 


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
dec_tree_reg = DecisionTreeRegressor()

param_distribution = dict(
    criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    max_depth = randint(1, 100)
)

# Next let's find best hyperparameters for Decision Tree
rscv = RandomizedSearchCV(dec_tree_reg, param_distributions=param_distribution, scoring='neg_mean_squared_error', n_iter=10, cv=5, random_state=42, verbose=2)
rscv.fit(X_train, y_train.values.ravel())  # Flatten y_train to 1D array

print(f"\nDecision Tree Results")
print(f"Best parameters: {rscv.best_params_}")
print(f"Best MSE: {-rscv.best_score_:.2f}")
print(f"Best RMSE: {np.sqrt(-rscv.best_score_):.2f}")

# Evaluate on test set
best_tree = rscv.best_estimator_
print(f"Test R² score: {best_tree.score(X_test, y_test):.4f}")


# Random Forest Regression (usually better than single Decision Tree)
print(f"\nRandom Forest Regression")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_train.values.ravel())

print(f"Training R² score: {rf_reg.score(X_train, y_train):.4f}")
print(f"Test R² score: {rf_reg.score(X_test, y_test):.4f}")

# Cross-validation
rf_cv_scores = cross_val_score(rf_reg, X_train, y_train.values.ravel(),
                                 scoring='neg_mean_squared_error', cv=5)
print(f"Cross-validation RMSE: {np.sqrt(-rf_cv_scores.mean()):.2f}")


# Final Comparison
print(f"\n{'='*50}")
print(f"Final Model Comparison")
print(f"Linear Regression Test R²:    {lin_reg.score(X_test, y_test):.4f}")
print(f"Ridge Regression Test R²:     {ridge_reg.score(X_test, y_test):.4f}")
print(f"Decision Tree Test R²:        {best_tree.score(X_test, y_test):.4f}")
print(f"Random Forest Test R²:        {rf_reg.score(X_test, y_test):.4f}")


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











