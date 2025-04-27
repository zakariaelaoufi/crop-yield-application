import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime as dt
from Functions import log_transformation

# Page configuration
st.set_page_config(page_title="Crop Yield Predictor", layout="wide")


# Load data and models
@st.cache_data
def load_data_and_models():
    """Cache data and model loading to improve performance"""
    cropyield = pd.read_csv('./data/yield_df.csv')
    model = pickle.load(open('./xgboost_model-final.sav', 'rb'))
    pipeline = pickle.load(open('./pipeline-final.sav', 'rb'))
    return cropyield, model, pipeline


cropyield, model, pipeline = load_data_and_models()

# Define the feature order for the model
feature_order = ['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year',
                 'avg_temp', 'pesticides_tonnes_log']

# App UI
st.title('Crop Yield Predictor')
st.write('Enter details below to predict crop yield')

with st.form(key='predict-crop-yield'):
    col1, col2 = st.columns(2)

    with col1:
        area = st.selectbox('Area', sorted(cropyield['Area'].unique()))
        item = st.selectbox('Item', sorted(cropyield['Item'].unique()))
        year = st.slider('Year', min_value=1900, max_value=dt.datetime.now().year,
                         value=dt.datetime.now().year, step=1)

    with col2:
        avg_rainfall = st.number_input('Average Rainfall (mm/year)', min_value=0.0)
        pesticides = st.number_input('Pesticides (tonnes)', min_value=0.0)
        avg_temp = st.number_input('Average Temperature (°C)', min_value=-20.0, max_value=50.0)

    submit_btn = st.form_submit_button(label='Predict Yield')

    if submit_btn:
        # Gather input data
        input_data = {
            'Area': area,
            'Item': item,
            'Year': year,
            'average_rain_fall_mm_per_year': avg_rainfall,
            'pesticides_tonnes_log': log_transformation(pesticides),
            'avg_temp': avg_temp
        }

        try:
            # Convert to DataFrame for consistent processing
            input_df = pd.DataFrame([input_data])

            # Ensure proper feature order
            input_df = input_df[feature_order]

            # Transform and predict
            with st.spinner('Predicting...'):
                transformed_data = pipeline.transform(input_df)
                prediction = model.predict(transformed_data)[0]
                exp_prediction = np.exp(prediction)

                # Display result
                st.success(f'Predicted Crop Yield: {exp_prediction:.2f} tonnes/hectare')

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please check your input values and try again.")

# Add information about the model
with st.expander("About this model"):
    st.write("""
    This application uses an XGBoost regression model to predict crop yields based on:
    - Geographical area
    - Crop type
    - Year
    - Average rainfall
    - Pesticide use
    - Average temperature

    The model was trained on historical [crop yield data](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset/data).
    """)