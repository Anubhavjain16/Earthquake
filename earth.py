import streamlit as st
import joblib
import numpy as np

# Load the trained XGBoost model
def load_model():
    return joblib.load('xgboost_earthquake_model.pkl')

xgb_model = load_model()

# Streamlit app setup
st.title("Earthquake Magnitude Prediction")

st.markdown("## Enter Earthquake Details")

# Input fields for user input
latitude = st.number_input("Latitude", value=0.0)
longitude = st.number_input("Longitude", value=0.0)
depth = st.number_input("Depth (km)", value=10.0)
mag_type = st.selectbox("Magnitude Type (encoded)", list(range(21)))
nst = st.number_input("Number of Stations (nst)", value=10.0)
gap = st.number_input("Gap", value=0.0)
dmin = st.number_input("Distance to Epicenter (dmin)", value=0.0)
rms = st.number_input("RMS", value=0.0)
horizontal_error = st.number_input("Horizontal Error", value=0.0)
depth_error = st.number_input("Depth Error", value=0.0)
mag_error = st.number_input("Magnitude Error", value=0.0)
mag_nst = st.number_input("Magnitude Nst", value=0.0)

# Create a feature array
features = np.array([[latitude, longitude, depth, mag_type, nst, gap, dmin, rms,
                      horizontal_error, depth_error, mag_error, mag_nst]])

# Predict button
if st.button("Predict"):
    # Perform prediction
    prediction = xgb_model.predict(features)[0]

    # Determine risk category
    if prediction < 4.0:
        risk = "Low Risk"
        color = "#2E8B57"  # Green
    elif 4.0 <= prediction < 6.0:
        risk = "Moderate Risk"
        color = "#1E3D59"  # Blue
    else:
        risk = "High Risk"
        color = "#B22222"  # Red

    # Display predicted magnitude with color-coded background
    st.markdown(
        f"<div style='background-color:{color}; padding:10px; border-radius:5px;'>"
        f"<h4 style='color:white;'>Predicted Magnitude: {prediction:.2f}</h4>"
        "</div>",
        unsafe_allow_html=True
    )

    # Display risk category
    st.markdown(
        f"<div style='background-color:{color}; padding:10px; border-radius:5px;'>"
        f"<h4 style='color:white;'>Risk Category: {risk}</h4>"
        "</div>",
        unsafe_allow_html=True
    )
