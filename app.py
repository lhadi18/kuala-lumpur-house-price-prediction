import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import gzip
import shutil
import os


# Function to decompress the model file
def decompress_model(input_file, output_file):
    try:
        with gzip.open(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    except Exception as e:
        st.error(f"Error decompressing the model file: {e}")
        raise


compressed_model_path = 'rf_model.joblib.gz'
decompressed_model_path = 'rf_model.joblib'

# Decompress the model file
if not os.path.exists(decompressed_model_path):
    decompress_model(compressed_model_path, decompressed_model_path)

# Load the trained model, scaler, feature columns, and data
try:
    model = joblib.load(decompressed_model_path)
except Exception as e:
    st.error(f"Error loading the decompressed model file: {e}")
    raise

scaler = joblib.load('scaler.joblib')
feature_columns = joblib.load('feature_columns.joblib')
initial_df = pd.read_csv('cleaned_data.csv')
encoded_df = pd.read_csv('encoded_data.csv')

# Extract unique locations, property types, and furnishing types for the dropdowns
locations = sorted([col.replace('Location_', '') for col in feature_columns if 'Location_' in col])
property_types = sorted([col.replace('Property Type_', '') for col in feature_columns if 'Property Type_' in col])
furnishing_types = sorted([col.replace('Furnishing_', '') for col in feature_columns if 'Furnishing_' in col])

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 24.1% 68.8%, rgb(50, 50, 50) 0%, rgb(0, 0, 0) 99.4%);
    }
    .stButton>button {
        background-color: green;
        color: white;
        font-weight: normal;
        border: none;
        outline: none;
    }
    .stButton>button:hover {
        background-color: darkgreen;
        color: white;
        font-weight: bold;
    }
    .stButton>button:focus {
        background-color: darkgreen;
        color: white;
        font-weight: bold;
        border: none;
        outline: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Kuala Lumpur House Price Prediction")

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Select Location", locations)

    rooms = st.number_input("Number of Rooms", min_value=1, max_value=20, value=2, step=1)

    car_parks = st.number_input("Number of Car Parks", min_value=0, max_value=10, value=1, step=1)

    size_unit = st.selectbox("Select Unit", ["Square Feet (sq. ft.)", "Square Meters (sq. m.)"])

with col2:
    property_type = st.selectbox("Property Type", property_types)

    bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=20, value=2, step=1)

    furnishing = st.selectbox("Furnishing", furnishing_types)

    size = st.number_input("Size", value=1500)

if size_unit == "Square Meters":
    size = size * 10.7639  # 1 square meter = 10.7639 square feet

input_data = {
    'Rooms': [rooms],
    'Bathrooms': [bathrooms],
    'Car Parks': [car_parks],
    'Size': [size],
    'Location_' + location: [1],
    'Property Type_' + property_type: [1],
    'Furnishing_' + furnishing: [1],
}

# Create a DataFrame and ensure it has the same columns as the model's training data
input_df = pd.DataFrame(input_data).reindex(columns=feature_columns, fill_value=0)

# Standardize the input data
input_scaled = scaler.transform(input_df)

adjust_for_inflation = st.checkbox("Adjust for Inflation")

if adjust_for_inflation:
    inflation_rate = st.number_input("Average Annual Inflation Rate (%)", min_value=0.0, max_value=100.0, value=3.0,
                                     step=0.1)
else:
    inflation_rate = None

dataset_upload_date = datetime(2019, 7, 4)
current_date = datetime.now()
years_since_upload = round((current_date - dataset_upload_date).days / 365.25)

# Make prediction
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)
    predicted_price = prediction[0]

    if adjust_for_inflation and inflation_rate is not None:
        inflation_factor = (1 + inflation_rate / 100) ** years_since_upload
        adjusted_price = predicted_price * inflation_factor
        st.success(f"Predicted Price: RM {predicted_price:,.2f}")
        st.success(f"Inflation-Adjusted Price: RM {adjusted_price:,.2f}")
    else:
        st.success(f"Predicted Price: RM {predicted_price:,.2f}")

st.markdown("**Note:** The dataset used to train this model was last updated on July 4, 2019.")
