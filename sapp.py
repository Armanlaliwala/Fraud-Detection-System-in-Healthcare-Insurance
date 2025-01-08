import streamlit as st
import pandas as pd
import pickle

# Load Saved Models and Scaler
@st.cache_resource
def load_models():
    with open('random_forest_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    with open('decision_tree_model.pkl', 'rb') as file:
        dt_model = pickle.load(file)
    with open('logistic_regression_model.pkl', 'rb') as file:
        lr_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return rf_model, dt_model, lr_model, scaler

rf_model, dt_model, lr_model, scaler = load_models()

# UI for the Fraud Detection System
st.title("Healthcare Insurance Fraud Detection")
st.markdown("### Identify fraudulent insurance claims effortlessly with machine learning.")

# Model Selection Dropdown
st.sidebar.title("Model Selection")
selected_model = st.sidebar.selectbox("Choose a Model:", 
                                       ("Random Forest", "Decision Tree", "Logistic Regression"))

# User Input Form
st.sidebar.title("Claim Data Input")
st.sidebar.markdown("Enter the details of the insurance claim:")

# Dropdown configuration for grouping fields
policy_options = {
    'Policy CSL 250/500': 'policy_csl_250/500',
    'Policy CSL 500/1000': 'policy_csl_500/1000'
}

education_options = {
    'College': 'insured_education_level_College',
    'High School': 'insured_education_level_High School',
    'JD': 'insured_education_level_JD',
    'MD': 'insured_education_level_MD',
    'Masters': 'insured_education_level_Masters',
    'PhD': 'insured_education_level_PhD'
}

occupation_options = {
    'Armed Forces': 'insured_occupation_armed-forces',
    'Craft Repair': 'insured_occupation_craft-repair',
    'Exec Managerial': 'insured_occupation_exec-managerial',
    'Farming Fishing': 'insured_occupation_farming-fishing',
    'Handlers Cleaners': 'insured_occupation_handlers-cleaners',
    'Machine Op Inspct': 'insured_occupation_machine-op-inspct',
    'Other Service': 'insured_occupation_other-service',
    'Private House Serv': 'insured_occupation_priv-house-serv',
    'Prof Specialty': 'insured_occupation_prof-specialty',
    'Protective Serv': 'insured_occupation_protective-serv',
    'Sales': 'insured_occupation_sales',
    'Tech Support': 'insured_occupation_tech-support',
    'Transport Moving': 'insured_occupation_transport-moving'
}

relationship_options = {
    'Not In Family': 'insured_relationship_not-in-family',
    'Other Relative': 'insured_relationship_other-relative',
    'Own Child': 'insured_relationship_own-child',
    'Unmarried': 'insured_relationship_unmarried',
    'Wife': 'insured_relationship_wife'
}

incident_options = {
    'Parked Car': 'incident_type_Parked Car',
    'Single Vehicle Collision': 'incident_type_Single Vehicle Collision',
    'Vehicle Theft': 'incident_type_Vehicle Theft'
}

collision_options = {
    'Rear Collision': 'collision_type_Rear Collision',
    'Side Collision': 'collision_type_Side Collision'
}

incident_severity_options = {
    'Minor Damage': 'incident_severity_Minor Damage',
    'Total Loss': 'incident_severity_Total Loss',
    'Trivial Damage': 'incident_severity_Trivial Damage'
}

authorities_options = {
    'Fire': 'authorities_contacted_Fire',
    'Other': 'authorities_contacted_Other',
    'Police': 'authorities_contacted_Police'
}

# Function to create dropdowns with checkboxes
def create_dropdown(label, options):
    selected_options = {}
    with st.sidebar.expander(label):
        for option, key in options.items():
            selected_options[key] = st.checkbox(option)
    return selected_options

# Collect user inputs
def user_input_features():
    # Numerical Inputs
    policy_annual_premium = st.sidebar.number_input('Policy Annual Premium', value=1000.0, format="%.2f")
    injury_claim = st.sidebar.number_input('Injury Claim Amount', value=500)
    property_claim = st.sidebar.number_input('Property Claim Amount', value=1000)
    months_as_customer = st.sidebar.number_input('Months as Customer', value=12)
    policy_deductable = st.sidebar.number_input('Policy Deductible', value=500)
    umbrella_limit = st.sidebar.number_input('Umbrella Limit', value=0)
    capital_gains = st.sidebar.number_input('Capital Gains', value=0)
    capital_loss = st.sidebar.number_input('Capital Loss', value=0)
    incident_hour_of_the_day = st.sidebar.slider('Incident Hour of the Day', 0, 23, 12)
    number_of_vehicles_involved = st.sidebar.number_input('Number of Vehicles Involved', value=1, min_value=1)
    bodily_injuries = st.sidebar.number_input('Number of Bodily Injuries', value=0, min_value=0)
    witnesses = st.sidebar.number_input('Number of Witnesses', value=0, min_value=0)
    vehicle_claim = st.sidebar.number_input('Vehicle Claim Amount', value=1000)

    # Categorical Inputs as Dropdowns and Checkboxes
    categorical_inputs = {}
    categorical_inputs.update(create_dropdown("Policy Options", policy_options))
    categorical_inputs.update(create_dropdown("Education Level", education_options))
    categorical_inputs.update(create_dropdown("Occupation", occupation_options))
    categorical_inputs.update(create_dropdown("Relationship", relationship_options))
    categorical_inputs.update(create_dropdown("Incident Type", incident_options))
    categorical_inputs.update(create_dropdown("Collision Type", collision_options))
    categorical_inputs.update(create_dropdown("Incident Severity", incident_severity_options))
    categorical_inputs.update(create_dropdown("Authorities Contacted", authorities_options))

    # Add standalone checkboxes
    categorical_inputs['insured_sex_MALE'] = st.sidebar.checkbox('Insured Sex: Male')
    categorical_inputs['property_damage_YES'] = st.sidebar.checkbox('Property Damage: Yes')
    categorical_inputs['police_report_available_YES'] = st.sidebar.checkbox('Police Report Available: Yes')

    # Prepare Data for Prediction
    data = {
        'policy_annual_premium': [policy_annual_premium],
        'injury_claim': [injury_claim],
        'property_claim': [property_claim],
        'months_as_customer': [months_as_customer],
        'policy_deductable': [policy_deductable],
        'umbrella_limit': [umbrella_limit],
        'capital-gains': [capital_gains],
        'capital-loss': [capital_loss],
        'incident_hour_of_the_day': [incident_hour_of_the_day],
        'number_of_vehicles_involved': [number_of_vehicles_involved],
        'bodily_injuries': [bodily_injuries],
        'witnesses': [witnesses],
        'vehicle_claim': [vehicle_claim],
    }

    # Add categorical inputs
    for key, value in categorical_inputs.items():
        data[key] = [int(value)]

    return pd.DataFrame(data)

input_df = user_input_features()

# Display Input Data
st.write("### Input Data")
st.dataframe(input_df)

# Preprocessing for prediction
def preprocess_input(input_df):
    # Add missing columns with default values
    for feature in features.keys():
        if feature not in input_df.columns:
            input_df[feature] = 0  # Add missing features with default value 0

    # Reorder the dataframe columns to match the model's training order
    input_df = input_df[features.keys()]
    return input_df

processed_input = preprocess_input(input_df)

# Prediction Logic
if st.button("Predict"):
    if selected_model == "Random Forest":
        prediction = rf_model.predict(processed_input)
    st.success(prediction)
