
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Define the filename for your pickled pipeline
model_filename = 'XGBoost_SKLearn_Pipeline_Final.pkl'

# Check if the model file exists
# Fixed: Used the string variable model_filename instead of a undefined variable name
if not os.path.exists(model_filename):
    st.error(f"Error: Model file '{model_filename}' not found.")
    st.stop() # Stop the app if the model is not found

# Load the trained pipeline
@st.cache_resource # Cache the model loading
def load_model(filename):
    try:
        with open(filename, 'rb') as f:
            pipeline = pickle.load(f)
        st.success("Model pipeline loaded successfully!")
        return pipeline
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        return None

pipeline = load_model(model_filename)

if pipeline is None:
    st.stop() # Stop if loading failed

st.title('Bike Sharing Demand Prediction')
st.write('Predict the count of total bike rentals based on historical data.')

# --- Streamlit UI Elements for Input ---

# Use sliders and selectboxes for user input corresponding to your model features
# Refer to the features used in your ColumnTransformer:
# Numeric: 'temp', 'humidity', 'windspeed', 'day', 'dayofyear', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos'
# Categorical: 'season', 'holiday', 'workingday', 'weather', 'hour_val', 'month_val', 'weekday_val', 'year_cat'

# Note: Some features like hour_sin, hour_cos, month_sin, month_cos, weekday_sin, weekday_cos,
# day, dayofyear will be calculated from user inputs hour, month, weekday, and datetime/year.

st.header('Input Features')

col1, col2, col3 = st.columns(3)

with col1:
    # datetime input to get year, month, day
    input_date = st.date_input('Date')
    # season, holiday, workingday (based on date and dataset info)
    # For simplicity, let's ask directly for these or derive them if possible
    # Deriving holiday/workingday from date is complex without a holiday calendar lookup.
    # Let's ask directly for simplicity in this example app.
    season = st.selectbox('Season', [1, 2, 3, 4], format_func=lambda x: {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}[x])
    holiday = st.selectbox('Is it a holiday?', [0, 1], format_func=lambda x: {0:'No', 1:'Yes'}[x])


with col2:
    input_time = st.time_input('Time')
    hour_val = input_time.hour # Extract hour
    st.write(f'Selected Hour: {hour_val}') # Display selected hour for clarity

    # Weather
    weather = st.selectbox('Weather', [1, 2, 3, 4], format_func=lambda x: {1:'Clear/Few clouds', 2:'Mist/Cloudy', 3:'Light Rain/Snow', 4:'Heavy Rain/Snow/Ice'}[x])
    # Workingday - derive from date and holiday
    # A simple derivation: not a holiday AND not Saturday (5) or Sunday (6)
    selected_date_weekday = input_date.weekday() # Monday is 0, Sunday is 6
    workingday = 0 if holiday == 1 or selected_date_weekday in [5, 6] else 1
    st.write(f'Is it a working day? {"Yes" if workingday == 1 else "No"}') # Display working day status


with col3:
    # Numerical features
    temp = st.slider('Temperature (°C)', -10.0, 40.0, 15.0)
    humidity = st.slider('Humidity (%)', 0, 100, 50)
    windspeed = st.slider('Windspeed (km/h)', 0.0, 60.0, 10.0)
    # The app assumes the user inputs raw values for humidity/windspeed; winsorizing will happen in the pipeline if included there.
    # If your pipeline does NOT include winsorizing, you might need to add it here before prediction.
    # Based on your notebook, winsorizing was applied BEFORE the ColumnTransformer, so it should NOT be in the pipeline.
    # Winsorizing should happen here before creating the input DataFrame.

# --- Manual Winsorizing (as done in the notebook before ColumnTransformer) ---
# Note: This requires knowing the winsorizing limits used during training.
# For robustness, ideally winsorizing limits would be part of the saved pipeline or configuration.
# For this example, we'll use the limits from the notebook, but be aware this is less robust.
def apply_manual_winsorizing(df):
    df_out = df.copy()
    # Apply winsorize_series_robust logic manually or re-implement here
    # Using hardcoded limits from your notebook (0.01 for humidity, 0.01 for humidity, 0.05 for windspeed)
    # A more robust approach would involve saving and loading these limits.
    # For demonstration, let's implement simple clipping based on training data percentiles (approximate)
    # WARNING: Hardcoding limits derived from training data percentiles is a simplification.
    # Proper implementation would save/load the winsorizing transformer or calculated limits.
    # Approx. values based on notebook:
    # Humidity lower limit (1st percentile) was likely > 0. Let's just clip the max for now.
    # Windspeed lower limit (5th percentile) is likely > 0. Let's clip max for now.
    # Clipping max based on rough observation from notebook EDA/describe:
    # Max humidity is 100, but 99th percentile might be lower.
    # Max windspeed can be high, 95th percentile might be around 35-40.
    # Let's re-evaluate based on actual winsorize output if possible, or stick to simple clipping for demo.
    # Simple clamping based on observed reasonable ranges from notebook EDA:
    # Note: The original notebook used (0.01, 0.01) for humidity and (0.05, 0.05) for windspeed.
    # This typically means clipping the lowest 1% and highest 1% of humidity, and lowest 5% and highest 5% of windspeed.
    # To replicate this accurately, you'd need the actual values corresponding to those percentiles from your training data.
    # Simple clipping to observed reasonable ranges is an approximation for this demo.
    df_out['humidity'] = np.clip(df_out['humidity'], 0, 100) # Humidity max is always 100
    # Based on notebook describe/EDA, a safe max for windspeed might be around 40-45
    df_out['windspeed'] = np.clip(df_out['windspeed'], 0, 45)

    # If you want to use the exact scipy winsorize, you'd need scipy installed and the original limits (or the transformer)
    # from scipy.stats.mstats import winsorize
    # Assuming you saved the actual limits or the winsorizer objects:
    # try:
    #    # Load saved winsorizer limits or transformer
    #    # For demo, using hardcoded values derived from your notebook's limits (requires re-calculation on training data)
    #    humidity_limits = (calculated_humidity_lower_percentile, calculated_humidity_upper_percentile)
    #    windspeed_limits = (calculated_windspeed_lower_percentile, calculated_windspeed_upper_percentile)
    #    df_out['humidity'] = winsorize(df_out['humidity'], limits=humidity_limits).data
    #    df_out['windspeed'] = winsorize(df_out['windspeed'], limits=windspeed_limits).data
    # except Exception as e:
    #    st.warning(f"Could not apply exact winsorizing: {e}. Using simple clipping.")
    #    df_out['humidity'] = np.clip(df_out['humidity'], 0, 100)
    #    df_out['windspeed'] = np.clip(df_out['windspeed'], 0, 45)

    return df_out


# --- Prediction Button ---
if st.button('Predict Bike Rentals'):
    if pipeline:
        # Prepare input data similar to how training data was prepared
        # Need to reconstruct the engineered features that were calculated BEFORE ColumnTransformer

        # 1. Create a DataFrame from user inputs
        input_data = {
            'season': [season],
            'holiday': [holiday],
            'workingday': [workingday], # Derived above
            'weather': [weather],
            'temp': [temp],
            'humidity': [humidity],
            'windspeed': [windspeed],
            # 'casual': [0], # Not available for prediction
            # 'registered': [0], # Not available for prediction
            # 'count': [0], # Target
            'datetime': [pd.to_datetime(f'{input_date} {input_time}')] # Combine date and time
        }
        input_df = pd.DataFrame(input_data)

        # 2. Apply manual winsorizing to humidity and windspeed
        # WARNING: This requires the same winsorizing logic/limits as training.
        input_df = apply_manual_winsorizing(input_df)


        # 3. Manually apply the feature engineering steps done BEFORE the ColumnTransformer
        # These include: extracting hour_val, month_val, weekday_val, day, year_cat, dayofyear,
        # and creating cyclical features hour_sin/cos, month_sin/cos, weekday_sin/cos.
        # Also dropping 'atemp'.

        # Replicate preprocess_initial_features logic
        if 'datetime' in input_df.columns:
            input_df['hour_val'] = input_df['datetime'].dt.hour
            input_df['month_val'] = input_df['datetime'].dt.month
            input_df['weekday_val'] = input_df['datetime'].dt.weekday # Monday=0, Sunday=6
            input_df['day'] = input_df['datetime'].dt.day
            input_df['year_cat'] = input_df['datetime'].dt.year.astype(str) # Year as category
            input_df['dayofyear'] = input_df['datetime'].dt.dayofyear
            input_df = input_df.drop('datetime', axis=1, errors='ignore') # Drop original datetime

        # Replicate create_cyclical_features logic
        if 'hour_val' in input_df.columns:
            input_df['hour_sin'] = np.sin(2 * np.pi * input_df['hour_val']/24.0)
            input_df['hour_cos'] = np.cos(2 * np.pi * input_df['hour_val']/24.0)
        if 'month_val' in input_df.columns:
            input_df['month_sin'] = np.sin(2 * np.pi * input_df['month_val']/12.0)
            input_df['month_cos'] = np.cos(2 * np.pi * input_df['month_val']/12.0)
        if 'weekday_val' in input_df.columns:
            input_df['weekday_sin'] = np.sin(2 * np.pi * input_df['weekday_val']/7.0)
            input_df['weekday_cos'] = np.cos(2 * np.pi * input_df['weekday_val']/7.0)

        # Ensure all columns expected by the pipeline's ColumnTransformer are present,
        # even if some were calculated. The ColumnTransformer's 'remainder'='drop' handles missing ones
        # from the *original* set not specified, but the *engineered* set must match the training cols before CT.
        # A robust approach is to ensure all engineered columns (numeric and categorical for OHE)
        # are present, potentially adding dummy columns if necessary (though less ideal).
        # Let's print the input_df columns before prediction to verify
        st.write("Input DataFrame before prediction:")
        st.write(input_df)
        st.write("Columns:", input_df.columns.tolist())

        # 4. Make prediction using the pipeline
        try:
            # The pipeline expects the input *after* the manual feature engineering (including cyclical and winsorizing).
            # The ColumnTransformer inside the pipeline will then handle scaling and OHE.
            predicted_log_count = pipeline.predict(input_df)

            # Inverse transform the prediction (from log scale back to original scale)
            predicted_count = np.expm1(predicted_log_count)

            # Ensure prediction is non-negative integer
            predicted_count = max(0, round(predicted_count[0])) # Use [0] as predict returns an array

            st.subheader('Predicted Bike Rentals')
            st.success(f'Predicted Total Count: **{predicted_count}**')

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Please check the input values and the model pipeline compatibility.")
            st.write("Input DataFrame structure:")
            st.write(input_df.head())
            # Attempt to get expected columns from the preprocessor for debugging
            try:
                ct = pipeline.named_steps['preprocessor']
                # This is a simplified way to get column names, might not be perfect for complex transformers
                expected_cols_after_ct = []
                for name, trans, cols in ct.transformers_:
                    if trans != 'drop':
                         if isinstance(cols, str): # Handle case with single column name string
                              expected_cols_after_ct.append(cols)
                         elif isinstance(cols, list): # Handle case with list of column names
                              expected_cols_after_ct.extend(cols)
                         # Note: OneHotEncoder outputs new column names. Getting the exact names after OHE is complex.
                         # For debugging, knowing the *input* columns to the CT is helpful.
                st.write("Expected *input* columns by the ColumnTransformer (approximate):", expected_cols_after_ct)
            except Exception as ct_e:
                 st.write(f"Could not retrieve expected columns from preprocessor: {ct_e}")


    else:
        st.warning("Model pipeline not loaded. Cannot make prediction.")

