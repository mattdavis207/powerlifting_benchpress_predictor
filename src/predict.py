import argparse
import joblib
import pandas as pd


# for creating age gr
def create_age_groups(age):
    if age < 14:
        return 'Youth'
    elif age <= 18:
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


# Parse command line arguments
parser = argparse.ArgumentParser(description='Predict bench press max based on squat, deadlift, and other attributes')
parser.add_argument("--bodyweight", type=float, required=True, help="Bodyweight in lbs")
parser.add_argument("--best-squat-lbs", type=float, required=True, help="Best squat in lbs")
parser.add_argument("--best-deadlift-lbs", type=float, required=True, help="Best deadlift in lbs")
parser.add_argument("--age", type=float, required=True, help="Age in years")
parser.add_argument("--sex", type=str, required=True, choices=['Male', 'Female'], help="Sex (Male or Female)")
parser.add_argument("--equipment-type", type=str, required=True,
                    choices=['Raw', 'Wraps', 'Single-ply', 'Multi-ply'],
                    help="Equipment type")

args = parser.parse_args()

# lbs to kgs conversion
LBS_TO_KG = 0.453592
best_squat_kg = args.best_squat_lbs * LBS_TO_KG
best_deadlift_kg = args.best_deadlift_lbs * LBS_TO_KG
bodyweight_kg = args.bodyweight * LBS_TO_KG

# create age group
age_group = create_age_groups(args.age)

# calculate engineered features same as in preprocess.py
squat_bw_ratio = best_squat_kg / bodyweight_kg
deadlift_bw_ratio = best_deadlift_kg / bodyweight_kg
squat_to_deadlift_ratio = best_squat_kg / best_deadlift_kg

# encode male or female
sex_binary = 1 if args.sex == 'Male' else 0

# create input dataframe matching the order from preprocess.py
input_data = pd.DataFrame({
    'Age': [args.age],
    'BodyweightKg': [bodyweight_kg],
    'BestSquatKg': [best_squat_kg],
    'BestDeadliftKg': [best_deadlift_kg],
    'SquatBWRatio': [squat_bw_ratio],
    'DeadliftBWRatio': [deadlift_bw_ratio],
    'SquatToDeadliftRatio': [squat_to_deadlift_ratio],
    'AgeGroup': [age_group],
    'Sex_Binary': [sex_binary],
    'Equipment': [args.equipment_type]
})

# Load the saved preprocessor and model
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/final_model.pkl")

# Apply the same preprocessing transformations
input_preprocessed = preprocessor.transform(input_data)

# Make prediction
prediction = model.predict(input_preprocessed)

# Convert prediction back to lbs for display
prediction_lbs = prediction[0] / LBS_TO_KG

print(f"Bench Press Prediction\n")
print("\n")
print(f"\nInput:")
print(f"  Age: {args.age} years ({age_group})")
print(f"  Sex: {args.sex}")
print(f"  Bodyweight: {bodyweight_kg:.1f} kg")
print(f"  Best Squat: {args.best_squat_lbs:.1f} lbs ({best_squat_kg:.1f} kg)")
print(f"  Best Deadlift: {args.best_deadlift_lbs:.1f} lbs ({best_deadlift_kg:.1f} kg)")
print(f"  Equipment: {args.equipment_type}")
print(f"\nPredicted Bench Press:")
print(f"  {prediction[0]:.1f} kg ({prediction_lbs:.1f} lbs)")