import pickle
import numpy as np

# Load the Logistic Regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Example input data as a dictionary
input_data = {
    "policy_annual_premium": 500.0,
    "injury_claim": 1,
    "property_claim": 0,
    "policy_csl_250/500": 1,
    "policy_csl_500/1000": 0,
    "insured_sex_MALE": 1,
    "insured_education_level_College": 1,
    "insured_education_level_High School": 0,
    "insured_education_level_JD": 0,
    "insured_education_level_MD": 0,
    "insured_education_level_Masters": 0,
    "insured_education_level_PhD": 0,
    "insured_occupation_armed-forces": 0,
    "insured_occupation_craft-repair": 1,
    "insured_occupation_exec-managerial": 0,
    "insured_occupation_farming-fishing": 0,
    "insured_occupation_handlers-cleaners": 0,
    "insured_occupation_machine-op-inspct": 0,
    "insured_occupation_other-service": 0,
    "insured_occupation_priv-house-serv": 0,
    "insured_occupation_prof-specialty": 0,
    "insured_occupation_protective-serv": 0,
    "insured_occupation_sales": 0,
    "insured_occupation_tech-support": 0,
    "insured_occupation_transport-moving": 0,
    "insured_relationship_not-in-family": 1,
    "insured_relationship_other-relative": 0,
    "insured_relationship_own-child": 0,
    "insured_relationship_unmarried": 0,
    "insured_relationship_wife": 0,
    "incident_type_Parked Car": 0,
    "incident_type_Single Vehicle Collision": 1,
    "incident_type_Vehicle Theft": 0,
    "collision_type_Rear Collision": 1,
    "collision_type_Side Collision": 0,
    "incident_severity_Minor Damage": 0,
    "incident_severity_Total Loss": 0,
    "incident_severity_Trivial Damage": 1,
    "authorities_contacted_Fire": 0,
    "authorities_contacted_Other": 0,
    "authorities_contacted_Police": 1,
    "property_damage_YES": 1,
    "police_report_available_YES": 1,
    "months_as_customer": 12,
    "policy_deductable": 500,
    "umbrella_limit": 1000000,
    "capital-gains": 1000,
    "capital-loss": 0,
    "incident_hour_of_the_day": 14,
    "number_of_vehicles_involved": 2,
    "bodily_injuries": 1,
    "witnesses": 0,
    "vehicle_claim": 1
}

# Convert the input data dictionary to a list of values (features)
input_values = list(input_data.values())

# Convert it to a NumPy array and reshape to match the expected input format for the model
input_array = np.array(input_values).reshape(1, -1)

# Scale the input data
scaled_input = scaler.transform(input_array)

# Predict using the Logistic Regression model
prediction = lr_model.predict(scaled_input)

print("Prediction:", prediction)
