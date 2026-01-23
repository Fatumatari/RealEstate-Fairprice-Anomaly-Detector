import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model pipeline
@st.cache_resource
def load_model():
    with open('models/xgb_pipeline.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Initialize the app
st.set_page_config(page_title="FairPrice Check", page_icon="🏠", layout="wide")

# Title and description
st.title("🏠 FairPrice Check: Real Estate Price Anomaly Detector")
st.markdown("""
This application helps you determine if a property listing is **underpriced**, **fairly priced**, or **overpriced** 
based on comparable properties in the same location.
""")

# Load model
try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Create input form
st.header("Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    state = st.selectbox("State/County", ["Nairobi", "Kiambu", "Kajiado", "Mombasa"])
    locality = st.text_input("Locality", placeholder="e.g., Westlands, Embakasi")
    category = st.selectbox("Listing Category", ["For Rent", "For Sale"])
    property_type = st.selectbox("Property Type", ["House", "Apartment"])
    
with col2:
    sub_type = st.selectbox("Sub Type", [
        "Bungalow", "Maisonette", "Townhouse", "Villa", "Mansion",
        "Detached Duplex", "Semi-Detached Duplex", "Studio Apartment",
        "Flat & Apartment", "Penthouse", "Bedsitter (Single Room)",
        "Block of Flats", "Missing"
    ])
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
    toilets = st.number_input("Toilets", min_value=0, max_value=10, value=2)

with col3:
    parking = st.number_input("Parking Spaces", min_value=0, max_value=10, value=1)
    furnished = st.selectbox("Furnished", [0, 1])
    serviced = st.selectbox("Serviced", [0, 1])
    shared = st.selectbox("Shared", [0, 1])
    price = st.number_input("Listed Price (KES)", min_value=0, value=50000)

# Prediction button
if st.button("🔍 Check Price Fairness", type="primary"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'toilets': [toilets],
        'parking': [parking],
        'furnished': [furnished],
        'serviced': [serviced],
        'shared': [shared],
        'type': [property_type],
        'sub_type': [sub_type],
        'state': [state],
        'category': [category],
        'bedrooms_raw': [bedrooms],
        'bathrooms_raw': [bathrooms],
        'parking_raw': [parking],
        'price_raw': [price]
    })
    
    # Add engineered features (placeholder values - in production, compute from training stats)
    input_data['price_position'] = 0
    input_data['price_per_bedroom'] = price / (bedrooms + 1)
    input_data['price_per_bathroom'] = price / (bathrooms + 1)
    input_data['bedroom_deviation'] = 0
    input_data['bathroom_deviation'] = 0
    input_data['location_density'] = 0.05
    
    try:
        # Make prediction
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[0]
        
        # Display results
        st.header("📊 Prediction Results")
        
        class_names = ['Underpriced', 'Fairly Priced', 'Overpriced']
        predicted_class = class_names[prediction[0]]
        
        # Color coding
        color_map = {
            'Underpriced': 'green',
            'Fairly Priced': 'blue',
            'Overpriced': 'red'
        }
        
        st.markdown(f"### Predicted Class: :{color_map[predicted_class]}[{predicted_class}]")
        
        # Show probabilities
        st.subheader("Confidence Scores")
        prob_col1, prob_col2, prob_col3 = st.columns(3)
        
        with prob_col1:
            st.metric("Underpriced", f"{probabilities[0]:.1%}")
        with prob_col2:
            st.metric("Fairly Priced", f"{probabilities[1]:.1%}")
        with prob_col3:
            st.metric("Overpriced", f"{probabilities[2]:.1%}")
        
        # Interpretation
        st.subheader("💡 Interpretation")
        if predicted_class == "Underpriced":
            st.success("This property appears to be priced below market rates. It may represent a good deal!")
        elif predicted_class == "Fairly Priced":
            st.info("This property is priced within normal market range for similar properties in this area.")
        else:
            st.warning("This property appears to be priced above market rates. Consider negotiating or looking for alternatives.")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Footer
st.markdown("---")
st.markdown("**Note:** This tool provides guidance based on historical data and should be used alongside professional real estate advice.")