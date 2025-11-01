import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("model.pkl")

st.title("🏠 Pakistan House Price Prediction App")
st.markdown("""
Enter the property details below to estimate its price.
""")

# --- Input fields ---
area_sqft = st.number_input("Area (sqft)", min_value=100, max_value=20000, step=100)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
baths = st.number_input("Number of Baths", min_value=1, max_value=10, step=1)

city = st.selectbox("City", ['Lahore', 'Karachi', 'Islamabad', 'Rawalpindi', 'Faisalabad'])
province = st.selectbox("Province", ['Punjab', 'Sindh', 'Islamabad Capital Territory'])
property_type = st.selectbox("Property Type", ['House', 'Flat', 'Upper Portion', 'Lower Portion', 'FarmHouse', 'Room'])
purpose = st.selectbox("Purpose", ['For Sale', 'For Rent'])

# --- Create input DataFrame ---
input_data = pd.DataFrame({
    'area_sqft': [area_sqft],
    'bedrooms': [bedrooms],
    'baths': [baths],
    'city': [city],
    'property_type': [property_type],
    'purpose': [purpose],
    'province_name': [province]
})

# --- Predict button ---
if st.button("Predict House Price"):
    prediction = model.predict(input_data)
    predicted_price = prediction[0]

    st.success(f"🏡 **Estimated Price:** PKR {predicted_price:,.0f}")
    st.balloons()
