import pandas as pd
import numpy as np 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector

X_train = pd.read_csv("../data/X_train.csv")
X_test = pd.read_csv("../data/X_test.csv")
y_train = pd.read_csv("../data/y_train.csv")
y_test = pd.read_csv("../data/y_test.csv")


# clean training data

# drop null entries by 'Age'
train_null_mask = X_train['Age'].notna() # creates boolean mask for non-null
X_train_clean = X_train[train_null_mask]
y_train_clean = y_train[train_null_mask]

test_null_mask = X_test['Age'].notna()
X_test_clean = X_test[test_null_mask]
y_test_clean = y_test[test_null_mask]

# remove name column (isn't useful)
X_train_clean.drop(columns=['Name'], inplace=True)
X_test_clean.drop(columns=['Name'], inplace=True)


# remove negative kg entry values

# training set
train_valid_mask = (
    (X_train_clean['BestDeadliftKg'] >= 0) &
    (X_train_clean['BestSquatKg'] >= 0) &
    (X_train_clean['BodyWeightKg'] >= 0) &
    (y_train_clean['BestBenchKg'] >= 0)
)

X_train_clean = X_train_clean[train_valid_mask]
y_train_clean = y_train_clean[train_valid_mask]

# test set 
test_valid_mask = (
    (X_test_clean['BestDeadliftKg'] >= 0) &
    (X_test_clean['BestSquatKg'] >= 0) &
    (X_test_clean['BodyWeightKg'] >= 0) &
    (y_test_clean['BestBenchKg'] >= 0)
)

X_test_clean = X_test_clean[test_valid_mask]
y_test_clean = y_test_clean[test_valid_mask]

# next feature engineering, lets add in the some ratios and weight classes/age groups

# bodyweight to lift ratios 

# training set
X_train_clean['SquatBWRatio'] = X_train_clean['BestSquatKg'] / X_train_clean['BodyWeightKg']
X_train_clean['DeadliftBWRatio'] = X_train_clean['BestDeadliftKg'] / X_train_clean['BodyWeightKg']

# test set
X_test_clean['SquatBWRatio'] = X_test_clean['BestSquatKg'] / X_test_clean['BodyWeightKg']
X_test_clean['DeadliftBWRatio'] = X_test_clean['BestDeadliftKg'] / X_test_clean['BodyWeightKg']

# next, add in age group bins
def create_age_groups(age):
    if age < 14:
        return 'Youth'
    elif age <=18:
        return 'Sub-Junior'
    elif age < 24:
        return 'Junior'
    elif 24 <= age < 40:
        return 'Open'
    elif 40 <= age < 50:
        return 'Master1'
    elif 50 <= age < 60:
        return 'Master2'
    elif 60 <= age < 70:
        return 'Master3'
    else:
        return 'Master4'

# apply to column
X_train_clean['AgeGroup'] = X_train_clean['age'].apply(create_age_groups)
X_test_clean['AgeGroup'] = X_test_clean['age'].apply(create_age_groups)

# IPF and USAPL official weight classes for male and female
# def create_weight_class_male(weight):
#     if weight <= 59:
#         return '59kg'
#     elif weight <= 66:
#         return '66kg'
#     elif weight <= 74:
#         return '74kg'
#     elif weight <= 83:
#         return '83kg'
#     elif weight <= 93:
#         return '93kg'
#     elif weight <= 105:
#         return '105kg'
#     elif weight <= 120:
#         return '120kg'
#     else:
#         return '120kg+'



# def create_weight_class_female(weight):
#     if weight <= 47:
#         return '47kg'
#     elif weight <= 52:
#         return '52kg'
#     elif weight <= 57:
#         return '57kg'
#     elif weight <= 63:
#         return '63kg'
#     elif weight <= 69:
#         return '69kg'
#     elif weight <= 76:
#         return '76kg'
#     elif weight <= 84:
#         return '84kg'
#     else:
#         return '84kg+'



# apply based on sex 
# X_train_clean['WeightClass'] = X_train_clean.apply( 
#     lambda row: create_weight_class_male(row['BodyweightKgs']) if row['Sex'] == 'Male' 
#     else create_weight_class_female(row['BodyweightKgs']), 
#     axis = 1
# )

# X_test_clean['WeightClass'] = X_test_clean.apply( 
#     lambda row: create_weight_class_male(row['BodyweightKgs']) if row['Sex'] == 'Male' 
#     else create_weight_class_female(row['BodyweightKgs']), 
#     axis = 1
# )


# finally, let's add a squat to deadlift ratio
X_train_clean['SquatToDeadliftRatio'] = X_train_clean['BestSquatKg'] / X_train_clean['BestDeadliftKg']
X_test_clean['SquatToDeadliftRatio'] = X_test_clean['BestSquatKg'] / X_test_clean['BestDeadliftKg']


# next, we will transform the features through scaling and encode the categorical features

# first encode Sex manually
X_train_clean['Sex_Binary'] = (X_train_clean['Sex'] == 'Male').astype(int)
X_test_clean['Sex_Binary'] = (X_test_clean['Sex'] == 'Male').astype(int)

# For Ordinal encoder
age_order = [['Youth', 'Sub-Junior', 'Junior', 'Open', 'Master1', 'Master2', 'Master3', 'Master4']]


num_pipeline = make_pipeline(StandardScaler())

preprocessor = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.float64)),
    ()

)





