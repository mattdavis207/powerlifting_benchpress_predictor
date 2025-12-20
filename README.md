# powerlifting_benchpress_predictor
Predict an athlete’s bench press one-rep max (1RM) using anthropometric and training variables, demonstrating an end-to-end supervised regression pipeline.

I used this dataset from Kaggle.com which is a powerlifting benchpress weight dataset: [https://www.kaggle.com/datasets/kukuroo3/powerlifting-benchpress-weight-predict?select=X_train.csv](https://www.kaggle.com/datasets/kukuroo3/powerlifting-benchpress-weight-predict?select=X_train.csv)


Models Trained:
* Linear Regression
* Ridge Regression
* Decision Tree
* Random Forest
* Gradient Boost
* XGBoost


The best model was XGBoost achieving a R^2 correlation score of 0.8738098740577698 and an RMSE of 18.98 kg.

## Engineering Part 
I created a packaged trained model and CLI interface for predicting benchpress weight.

## Insights/Problems
* A lot of the instances were biased heavily towards certain ages and other features which may skew results. (Ex. Unrepresentative of ages below 23 or above 40)
* The features that weighed the most importance included the best squat and deadlift weight. (0.6165, 0.2829 respectively)
