import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Real Estate Investment Predictor",
    page_icon="🏠",
    layout="wide"
)


# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('D:/real_estate_investment_predictor/model/final_model.pkl')


# Load the scaler (if available)
@st.cache_resource
def load_scaler():
    try:
        return joblib.load('D:/real_estate_investment_predictor/model/scaler.pkl')
    except:
        st.warning("Scaler not found. Using raw features for prediction.")
        return None


try:
    model = load_model()
    scaler = load_scaler()
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.info("Please ensure the model file exists in the correct path.")
    model = None

# Title and description
st.title("🏠 Real Estate Investment Predictor")
st.markdown("""
This app predicts whether a real estate property is a safe investment based on various factors.
Enter the property details below to get a prediction.
""")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    price_growth = st.slider("Price Growth 5Y (%)", min_value=-10.0, max_value=30.0, value=5.0, step=0.1)
    crime_rate = st.slider("Crime Rate (per 1000)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
    rental_yield = st.slider("Rental Yield (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
    population_growth = st.slider("Population Growth (%)", min_value=-5.0, max_value=10.0, value=1.5, step=0.1)
    unemployment_rate = st.slider("Unemployment Rate (%)", min_value=0.0, max_value=25.0, value=5.0, step=0.1)

with col2:
    schools_nearby = st.slider("Schools Nearby (count)", min_value=0, max_value=20, value=3, step=1)
    hospitals_nearby = st.slider("Hospitals Nearby (count)", min_value=0, max_value=10, value=2, step=1)
    infrastructure_projects = st.slider("Infrastructure Projects", min_value=0, max_value=10, value=6, step=1)
    natural_disaster_risk = st.slider("Natural Disaster Risk (0-10)", min_value=0, max_value=10, value=3, step=1)
    market_demand_index = st.slider("Market Demand Index", min_value=0, max_value=10, value=7, step=1)

# Create a prediction button
if st.button("Predict Investment Safety", type="primary"):
    # Prepare the input data - using EXACT column names from training dataset
    input_data = pd.DataFrame({
        'Price_Growth_5Y(%)': [price_growth],
        'Crime_Rate(per_1000)': [crime_rate],
        'Rental_Yield(%)': [rental_yield],
        'Population_Growth(%)': [population_growth],
        'Unemployment_Rate(%)': [unemployment_rate],
        'Schools_Nearby': [schools_nearby],
        'Hospitals_Nearby': [hospitals_nearby],
        'Infrastructure_Projects': [infrastructure_projects],
        'Natural_Disaster_Risk(0-10)': [natural_disaster_risk],
        'Market_Demand_Index': [market_demand_index]
    })

    try:
        # Scale the input data if scaler is available
        if scaler is not None:
            input_processed = scaler.transform(input_data)
        else:
            input_processed = input_data.values

        # Make prediction
        prediction = model.predict(input_processed)
        prediction_proba = model.predict_proba(input_processed)

        # Display prediction
        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success("✅ This appears to be a SAFE investment!")
            safe_proba = prediction_proba[0][1] * 100
            st.metric("Confidence", f"{safe_proba:.2f}%")
        else:
            st.error("❌ This appears to be a RISKY investment!")
            risky_proba = prediction_proba[0][0] * 100
            st.metric("Confidence", f"{risky_proba:.2f}%")

        # Feature importance visualization (if available)
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': input_data.columns.tolist(),
                'Importance': model.feature_importances_
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)

            st.bar_chart(importance_df.set_index('Feature'))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Debug info: Input data shape: " + str(input_data.shape))
        st.info("Debug info: Feature names: " + ", ".join(input_data.columns.tolist()))

        # Show more detailed error information for debugging
        st.expander("Technical Details").write(f"Error type: {type(e).__name__}\nError details: {str(e)}")

        # Check if model has feature_names_in_ attribute (scikit-learn >= 1.0)
        if hasattr(model, 'feature_names_in_'):
            st.info("Model expects these features: " + ", ".join(model.feature_names_in_))

    # Add this section to your app.py after making a prediction
    if prediction is not None:
        # Feature importance visualization (if available)
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': input_data.columns.tolist(),
                'Importance': model.feature_importances_
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)

            # Create a horizontal bar chart for better visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance in Prediction')

            # Add values to the end of each bar
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width
                ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.3f}',
                        va='center', ha='left', fontsize=9)

            st.pyplot(fig)

            # Also show as a table for precise values
            st.write("Feature Importance Values:")
            st.dataframe(importance_df.style.format({'Importance': '{:.4f}'}))
# Add a section for recommendations
st.subheader("How to interpret the results")
st.markdown("""
- A **SAFE** investment typically has good price growth, low crime rates, and manageable natural disaster risk.
- A **RISKY** investment may have poor growth potential or high risk factors that could affect your return.

Remember that this model should be used as one of many tools in your investment decision process.
""")

# Add information about the model
with st.expander("About the Model"):
    st.markdown("""
    This prediction model was built using a Random Forest Classifier trained on historical real estate data.

    **Key features the model considers:**
    - Price growth over 5 years
    - Crime rates in the area
    - Rental yield potential
    - Population growth trends
    - Unemployment rates
    - Proximity to essential services (schools, hospitals)
    - Infrastructure projects in the area
    - Natural disaster risk assessment
    - Market demand index

    The model was trained with scikit-learn and achieved strong performance metrics including balanced precision and recall.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Real Estate Investment Predictor | Built with Streamlit")
