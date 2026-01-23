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
    category = st.selectbox("Listing Category", ["For Rent", "For Sale"])
    property_type = st.selectbox("Property Type", ["House", "Apartment"])
    sub_type = st.selectbox("Sub Type", [
        "Bungalow", "Maisonette", "Townhouse", "Villa", "Mansion",
        "Detached Duplex", "Semi-Detached Duplex", "Studio Apartment",
        "Flat & Apartment", "Penthouse", "Bedsitter (Single Room)",
        "Block of Flats", "Missing"
    ])

with col2:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2, step=1)
    toilets = st.number_input("Toilets", min_value=0, max_value=10, value=2, step=1)
    parking = st.number_input("Parking Spaces", min_value=0, max_value=10, value=1, step=1)

with col3:
    furnished = st.selectbox("Furnished", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    serviced = st.selectbox("Serviced", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    shared = st.selectbox("Shared", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    price_qualifier = st.selectbox("Price Qualifier", ["per month", "Sale", "per annum"])

# Prediction button
if st.button("🔍 Check Price Fairness", type="primary"):
    
    # Create input dataframe with all required features
    # These are the features expected by your trained pipeline
    input_data = pd.DataFrame({
        'bedrooms': [int(bedrooms)],
        'bathrooms': [int(bathrooms)],
        'toilets': [int(toilets)],
        'parking': [int(parking)],
        'furnished': [int(furnished)],
        'serviced': [int(serviced)],
        'shared': [int(shared)],
        'type': [property_type],
        'sub_type': [sub_type],
        'state': [state],
        'category': [category],
        'price_qualifier': [price_qualifier],
        'bedrooms_raw': [float(bedrooms)],
        'bathrooms_raw': [float(bathrooms)],
        'parking_raw': [float(parking)],
        # Engineered features - these will be computed by the pipeline
        'price_position': [0.0],
        'price_per_bedroom': [0.0],
        'price_per_bathroom': [0.0],
        'bedroom_deviation': [0.0],
        'bathroom_deviation': [0.0],
        'location_density': [0.05]
    })
    
    try:
        # Make prediction using the pipeline
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
            st.success("✅ This property appears to be priced below market rates. It may represent a good deal!")
        elif predicted_class == "Fairly Priced":
            st.info("ℹ️ This property is priced within normal market range for similar properties in this area.")
        else:
            st.warning("⚠️ This property appears to be priced above market rates. Consider negotiating or looking for alternatives.")
        
        # Show probability distribution chart
        st.subheader("📈 Probability Distribution")
        prob_df = pd.DataFrame({
            'Class': class_names,
            'Probability': probabilities
        })
        st.bar_chart(prob_df.set_index('Class'))
            
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
        st.error("Please ensure all fields are filled correctly.")
        st.exception(e)  # Show detailed error for debugging

# Footer
st.markdown("---")
st.markdown("""
**Note:** This tool provides guidance based on historical data and should be used alongside professional real estate advice.

**About the Model:** This prediction is based on an XGBoost classifier trained on real estate data from Kenya.
""")