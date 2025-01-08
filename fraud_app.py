import streamlit as st
import numpy as np
import pickle

# Load actual models and scaler (replace MockModel and MockScaler with actual ones)
decision_tree_model = pickle.load(open('decision_tree_model.pkl', 'rb'))
random_forest_model = pickle.load(open('random_forest_model.pkl', 'rb'))
logistic_regression_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define features and their data types
features = {
    "policy_annual_premium": "float",
    "injury_claim": "int",
    "property_claim": "int",
    "policy_csl_250/500": "bool",
    "policy_csl_500/1000": "bool",
    "insured_sex_MALE": "bool",
    "insured_education_level_College": "bool",
    "insured_education_level_High School": "bool",
    "insured_education_level_JD": "bool",
    "insured_education_level_MD": "bool",
    "insured_education_level_Masters": "bool",
    "insured_education_level_PhD": "bool",
    "insured_occupation_armed-forces": "bool",
    "insured_occupation_craft-repair": "bool",
    "insured_occupation_exec-managerial": "bool",
    "insured_occupation_farming-fishing": "bool",
    "insured_occupation_handlers-cleaners": "bool",
    "insured_occupation_machine-op-inspct": "bool",
    "insured_occupation_other-service": "bool",
    "insured_occupation_priv-house-serv": "bool",
    "insured_occupation_prof-specialty": "bool",
    "insured_occupation_protective-serv": "bool",
    "insured_occupation_sales": "bool",
    "insured_occupation_tech-support": "bool",
    "insured_occupation_transport-moving": "bool",
    "insured_relationship_not-in-family": "bool",
    "insured_relationship_other-relative": "bool",
    "insured_relationship_own-child": "bool",
    "insured_relationship_unmarried": "bool",
    "insured_relationship_wife": "bool",
    "incident_type_Parked Car": "bool",
    "incident_type_Single Vehicle Collision": "bool",
    "incident_type_Vehicle Theft": "bool",
    "collision_type_Rear Collision": "bool",
    "collision_type_Side Collision": "bool",
    "incident_severity_Minor Damage": "bool",
    "incident_severity_Total Loss": "bool",
    "incident_severity_Trivial Damage": "bool",
    "authorities_contacted_Fire": "bool",
    "authorities_contacted_Other": "bool",
    "authorities_contacted_Police": "bool",
    "property_damage_YES": "bool",
    "police_report_available_YES": "bool",
    "months_as_customer": "int",
    "policy_deductable": "int",
    "umbrella_limit": "int",
    "capital-gains": "int",
    "capital-loss": "int",
    "incident_hour_of_the_day": "int",
    "number_of_vehicles_involved": "int",
    "bodily_injuries": "int",
    "witnesses": "int",
    "vehicle_claim": "int",
}

# Preprocess user input
def preprocess_input(user_input):
    input_data = []
    for feature, dtype in features.items():
        # Use default values if no input is provided
        if dtype == "bool":
            input_data.append(1 if user_input[feature] == "True" else 0)
        elif dtype == "int":
            input_data.append(int(user_input.get(feature, 0)))  # Default to 0
        elif dtype == "float":
            input_data.append(float(user_input.get(feature, 0.0)))  # Default to 0.0
    return np.array(input_data).reshape(1, -1)

# Prediction function
def predict_fraud(input_data, model):
    # If all input data is zero, return "Fraud"
    if np.all(input_data == 0):
        return "Fraud"
    try:
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        return "Fraud" if prediction[0] == 1 else "Not Fraud"
    except Exception as e:
        # Handle any unexpected errors and default to "Fraud"
        return "Fraud"

# Streamlit App
st.title("Fraud Detection System")
st.header("Enter Details for Prediction")

# Dropdown to select the model
model_choice = st.selectbox(
    "Select the Model:",
    ("Decision Tree", "Random Forest", "Logistic Regression")
)

# Map selected model to actual model object
if model_choice == "Decision Tree":
    selected_model = decision_tree_model
elif model_choice == "Random Forest":
    selected_model = random_forest_model
else:
    selected_model = logistic_regression_model

user_input = {}

# Collect inputs for all features dynamically
for feature, dtype in features.items():
    if dtype == "bool":
        user_input[feature] = st.selectbox(f"{feature} (False/True):", [ "False", "True"])
    elif dtype == "int":
        user_input[feature] = st.number_input(f"{feature} (Integer):", value=0, step=1)
    elif dtype == "float":
        user_input[feature] = st.number_input(f"{feature} (Float):", value=0.0, step=0.1)

# Predict Button
if st.button("Predict"):
    input_data = preprocess_input(user_input)
    prediction_result = predict_fraud(input_data, selected_model)
    st.success(f"Prediction: {prediction_result}")
