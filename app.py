import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="FairPrice Check - Real Estate Analyzer",
    page_icon="🏠",
    layout="wide"
)

# Load the trained model and statistics
@st.cache_resource
def load_model_and_stats():
    try:
        with open('models/xgb_pipeline.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/training_stats.pkl', 'rb') as f:
            stats = pickle.load(f)
        
        with open('models/state_locality_mapping.pkl', 'rb') as f:
            state_locality_map = pickle.load(f)
        
        return model, stats, state_locality_map
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found. Please ensure models are trained and saved in the 'models' folder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, stats, state_locality_map = load_model_and_stats()

# Convert location_price_stats back to DataFrame
location_stats_df = pd.DataFrame(stats['location_price_stats'])

# Title and description
st.title("🏠 FairPrice Check: Real Estate Price Analyzer")
st.markdown("""
This tool helps you determine if a property is **underpriced**, **fairly priced**, or **overpriced** 
based on comparable listings in the same locality.
""")

# Sidebar for inputs
st.sidebar.header("📋 Property Details")

# Location inputs
state = st.sidebar.selectbox(
    "State/County *",
    options=sorted(stats['valid_states'])
)

# Filter localities based on selected state
available_localities = state_locality_map.get(state, [])

locality = st.sidebar.selectbox(
    "Locality *",
    options=available_localities
)

# Property classification
st.sidebar.subheader("Property Type")

category = st.sidebar.selectbox(
    "Listing Category *",
    options=['For Rent', 'For Sale']
)

property_type = st.sidebar.selectbox(
    "Property Type *",
    options=['House', 'Apartment']
)

sub_type = st.sidebar.selectbox(
    "Property Sub-Type *",
    options=[
        'Bungalow', 'Townhouse', 'Mansion', 'Maisonette', 
        'Villa', 'Detached Duplex', 'Semi-Detached Duplex',
        'Flat & Apartment', 'Studio Apartment', 'Penthouse',
        'Bedsitter (Single Room)', 'Block of Flats', 'Missing'
    ]
)

# Property features
st.sidebar.subheader("Property Features")

bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, max_value=10, value=2)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=0, max_value=10, value=2)
toilets = st.sidebar.number_input("Toilets", min_value=0, max_value=10, value=2)
parking = st.sidebar.number_input("Parking Spaces", min_value=0, max_value=10, value=1)

# Amenities
st.sidebar.subheader("Amenities")
furnished = st.sidebar.checkbox("Furnished")
serviced = st.sidebar.checkbox("Serviced")
shared = st.sidebar.checkbox("Shared")

# Price input
st.sidebar.subheader("Pricing")
price = st.sidebar.number_input(
    "Listed Price (KES) *",
    min_value=1000,
    max_value=500000000,
    value=50000,
    step=5000,
    help="Enter the listed price in Kenyan Shillings"
)

# Main content area - show instructions before prediction
if 'prediction_made' not in st.session_state:
    st.info("""
    👈 **Get Started:**
    1. Enter property details in the sidebar
    2. Click "Analyze Price" to get results
    3. View pricing classification and market insights
    """)
    
    st.markdown("### 📊 How It Works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🟢 Underpriced")
        st.write("Property priced below 25th percentile of similar listings in the area")
    
    with col2:
        st.markdown("#### 🟡 Fairly Priced")
        st.write("Property priced between 25th and 75th percentile (middle 50%)")
    
    with col3:
        st.markdown("#### 🔴 Overpriced")
        st.write("Property priced above 75th percentile of similar listings in the area")

# Predict button
if st.sidebar.button("🔍 Analyze Price", type="primary", use_container_width=True):
    
    # Get location stats for the selected locality
    loc_stats = location_stats_df[location_stats_df['locality'] == locality]
    
    if loc_stats.empty:
        st.error(f"❌ No pricing data available for **{locality}** in our training data.")
        st.warning("💡 Try selecting a different locality with more historical listings.")
        st.stop()
    
    # Extract location statistics
    loc_q25 = loc_stats['loc_q25'].values[0]
    loc_median = loc_stats['loc_median'].values[0]
    loc_q75 = loc_stats['loc_q75'].values[0]
    
    # Feature engineering (same as training)
    price_position = (price - loc_median) / loc_median if loc_median != 0 else 0
    price_per_bedroom = price / (bedrooms + 1)
    price_per_bathroom = price / (bathrooms + 1)
    
    # Get location averages
    loc_avg_bedrooms = stats['bedroom_mean']
    loc_avg_bathrooms = stats['bathroom_mean']
    
    bedroom_deviation = bedrooms - loc_avg_bedrooms
    bathroom_deviation = bathrooms - loc_avg_bathrooms
    
    location_density = stats['location_density'].get(locality, 0.01)
    
    # Create input dataframe matching training feature order
    # Model expects these exact columns in this order (from feature_names_in_):
    # ['id', 'price_qualifier', 'bedrooms', 'bathrooms', 'toilets', 'furnished',
    #  'serviced', 'shared', 'parking', 'category', 'type', 'sub_type', 'state',
    #  'sub_locality', 'listdate', 'bedrooms_raw', 'bathrooms_raw', 'parking_raw',
    #  'price_position', 'price_per_bedroom', 'price_per_bathroom',
    #  'bedroom_deviation', 'bathroom_deviation', 'location_density']
    
    input_data = pd.DataFrame({
        'id': [0],
        'price_qualifier': [''],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'toilets': [toilets],
        'furnished': [1 if furnished else 0],
        'serviced': [1 if serviced else 0],
        'shared': [1 if shared else 0],
        'parking': [parking],
        'category': [category],
        'type': [property_type],
        'sub_type': [sub_type],
        'state': [state],
        'sub_locality': [''],
        'listdate': ['2020-01-01'],  # Placeholder date as string
        'bedrooms_raw': [bedrooms],
        'bathrooms_raw': [bathrooms],
        'parking_raw': [parking],
        'price_position': [price_position],
        'price_per_bedroom': [price_per_bedroom],
        'price_per_bathroom': [price_per_bathroom],
        'bedroom_deviation': [bedroom_deviation],
        'bathroom_deviation': [bathroom_deviation],
        'location_density': [location_density]
    })
    
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Mark prediction as made
        st.session_state.prediction_made = True
        
        # Display results
        st.success("✅ Analysis Complete!")
        st.header("📊 Price Analysis Results")
        
        # Classification result with color coding
        labels = {0: "🟢 Underpriced", 1: "🟡 Fairly Priced", 2: "🔴 Overpriced"}
        
        # Large metric display
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            st.metric("Price Classification", labels[prediction])
        
        with col2:
            st.metric("Listed Price", f"KES {price:,.0f}")
        
        with col3:
            deviation = ((price - loc_median) / loc_median) * 100
            st.metric("vs Market Median", f"{deviation:+.1f}%")
        
        # Confidence scores
        st.subheader("📈 Confidence Scores")
        
        conf_col1, conf_col2, conf_col3 = st.columns(3)
        
        with conf_col1:
            st.metric("🟢 Underpriced", f"{probabilities[0]*100:.1f}%")
        
        with conf_col2:
            st.metric("🟡 Fairly Priced", f"{probabilities[1]*100:.1f}%")
        
        with conf_col3:
            st.metric("🔴 Overpriced", f"{probabilities[2]*100:.1f}%")
        
        # Market context
        st.subheader("📍 Market Context for " + locality)
        
        context_col1, context_col2 = st.columns(2)
        
        with context_col1:
            st.markdown("**Price Distribution:**")
            st.write(f"- Lower 25% (Q1): KES {loc_q25:,.0f}")
            st.write(f"- Median (Q2): KES {loc_median:,.0f}")
            st.write(f"- Upper 75% (Q3): KES {loc_q75:,.0f}")
        
        with context_col2:
            st.markdown("**Your Listing:**")
            if price < loc_q25:
                st.write(f"✅ **{((loc_q25 - price) / price * 100):.1f}%** below lower quartile")
            elif price > loc_q75:
                st.write(f"⚠️ **{((price - loc_q75) / loc_q75 * 100):.1f}%** above upper quartile")
            else:
                st.write(f"✓ Within normal market range")
            
            st.write(f"Market density: {location_density*100:.2f}% of listings")
        
        # Interpretation and recommendations
        st.subheader("💡 What This Means")
        
        if prediction == 0:
            st.success("""
            ### ✅ Great Deal Detected!
            
            This property is priced **below typical market rates** for this area.
            
            **Recommended Actions:**
            - ✓ Consider viewing the property soon
            - ✓ Verify the property condition matches the listing
            - ✓ Check for any hidden issues that might explain the low price
            - ✓ Act quickly as underpriced properties move fast
            """)
            
        elif prediction == 1:
            st.info("""
            ### ℹ️ Fair Market Price
            
            This listing is priced **within normal market expectations** for similar properties in this area.
            
            **Recommended Actions:**
            - ✓ Price is competitive and reasonable
            - ✓ Good baseline for negotiations
            - ✓ Compare amenities with other listings
            - ✓ Standard due diligence applies
            """)
            
        else:
            st.warning("""
            ### ⚠️ Above Market Price
            
            This property is priced **higher than typical listings** in this area.
            
            **Recommended Actions:**
            - ⚠️ Consider negotiating for a lower price
            - ⚠️ Ensure premium features justify the price
            - ⚠️ Compare with similar properties in the area
            - ⚠️ Ask seller about pricing rationale
            """)
        
        # Additional property insights
        with st.expander("🔍 See Detailed Property Analysis"):
            st.markdown("### Property Characteristics vs Market Average")
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.write("**Size Comparison:**")
                st.write(f"- Bedrooms: {bedrooms} (Avg: {loc_avg_bedrooms:.1f})")
                st.write(f"- Bathrooms: {bathrooms} (Avg: {loc_avg_bathrooms:.1f})")
                st.write(f"- Parking: {parking} spaces")
            
            with insight_col2:
                st.write("**Amenities:**")
                st.write(f"- Furnished: {'✅ Yes' if furnished else '❌ No'}")
                st.write(f"- Serviced: {'✅ Yes' if serviced else '❌ No'}")
                st.write(f"- Shared: {'✅ Yes' if shared else '❌ No'}")
            
            st.markdown("### Pricing Metrics")
            st.write(f"- Price per bedroom: KES {price_per_bedroom:,.0f}")
            st.write(f"- Price per bathroom: KES {price_per_bathroom:,.0f}")
            st.write(f"- Price position: {price_position:.2%} {'above' if price_position > 0 else 'below'} median")
            
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
        st.info("Please check that all required fields are filled correctly.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>⚠️ Disclaimer:</strong> This tool provides guidance based on historical market data and machine learning analysis. 
    Always conduct your own research, property inspection, and consult with real estate professionals before making decisions.</p>
    <p style='color: gray; font-size: 0.9em;'>Powered by XGBoost | Data: Kenya Property Centre</p>
</div>
""", unsafe_allow_html=True)