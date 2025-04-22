import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib

from make_dataset import create_features
from naive_approach import get_column_specs, prepare_data, zeroshot_eval, simple_diagonal_averaging
from ml_approach import format_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTEXT_LENGTH = 52
PREDICTION_LENGTH = 6

# Custom theme settings
st.set_page_config(
    page_title="Glucose Level Prediction App",
    page_icon="📊",
    layout="wide"
)

# Apply custom styling with CSS
st.markdown("""
<style>
    /* Primary accent color */
    .stButton button, .stSelectbox, .stMultiselect, .stSlider, .stNumberInput {
        border-color: #58A618 !important;
    }
    .stProgress .st-bo {
        background-color: #58A618 !important;
    }
    .st-bq {
        color: #58A618 !important;
    }
    /* Header styling */
    h1, h2, h3 {
        color: #58A618 !important;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #58A618 !important;
    }
    /* Success messages */
    .element-container .stAlert.st-ae.st-af {
        border-color: #58A618 !important;
        color: #58A618 !important;
    }
    /* Link color */
    a {
        color: #58A618 !important;
    }
    /* Button color */
    .stButton>button {
        background-color: #58A618 !important;
        color: white !important;
    }
    /* Make background white */
    .stApp {
        background-color: white !important;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f0f0;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        border: 1px solid #ccc;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: white;
        border-bottom: 3px solid #58A618;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'combined_data' not in st.session_state:
    st.session_state.combined_data = None
if 'files_uploaded' not in st.session_state:
    st.session_state.files_uploaded = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False

# Title and description
st.title("Glucose Level Prediction App")
st.markdown("""
This app allows you to upload glucose measurements, food logs, and accelerometer data 
to analyze patterns and predict glucose levels.
""")

# Choose data source
st.subheader("Choose Data Source")
data_option = st.selectbox(
    "Select how you'd like to provide input data:",
    ("Upload files", "Sample A", "Sample B")
)

glucose_data = None
food_data = None
accel_data = None
combined_data = None
show_tabs = False

if data_option == "Upload files":
    st.subheader("Upload Your Data Files")

    glucose_file = st.file_uploader("Upload Glucose Levels CSV", type=["csv"], key="glucose")
    food_file = st.file_uploader("Upload Food Logs CSV", type=["csv"], key="food")
    accel_file = st.file_uploader("Upload Accelerometer Data CSV", type=["csv"], key="accel")
    
    st.subheader("Patient Demographics")

    # Gender selection
    gender = st.selectbox("Select Patient Gender", options=["Female", "Male", "Other"], index=0)

    # HbA1c input
    hba1c = st.number_input("Enter HbA1c (%)", min_value=3.0, max_value=15.0, step=0.1)

    all_files_uploaded = (glucose_file is not None) and (food_file is not None) and (accel_file is not None)

    # Attempt to load files if they exist
    if glucose_file is not None:
        try:
            glucose_data = pd.read_csv(glucose_file)
            st.success("Glucose data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading glucose data: {e}")
            glucose_data = None

    if food_file is not None:
        try:
            food_data = pd.read_csv(food_file)
            st.success("Food logs loaded successfully!")
        except Exception as e:
            st.error(f"Error loading food logs: {e}")
            food_data = None

    if accel_file is not None:
        try:
            accel_data = pd.read_csv(accel_file)
            st.success("Accelerometer data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading accelerometer data: {e}")
            accel_data = None
    
    # Update the upload status in session state
    st.session_state.files_uploaded = all_files_uploaded
    
    # Show message if not all files are uploaded
    if not all_files_uploaded:
        st.warning("Please upload all three data files to enable data processing.")
    
    col1, col2, col3 = st.columns([1,1,1])

    with col2:
        # Add a button to process the data - disabled until all files are uploaded
        if st.button('Process Data', key='process_data_button', disabled=not all_files_uploaded):
            if all_files_uploaded:
                try:
                    # Call create_features with appropriate parameters
                    combined_data = create_features(
                        bg_df=glucose_data,
                        food_df=food_data,
                        acc_df=accel_data,
                        gender=gender,
                        hba1c=hba1c,
                        add_patient_id=True
                    )
                    st.session_state.combined_data = combined_data
                    st.session_state.data_processed = True
                    st.success("Data processed successfully!")
                    show_tabs = True
                except Exception as e:
                    st.error(f"Error processing data: {e}")
                    st.session_state.data_processed = False
                    show_tabs = False

    st.subheader("Expected File Formats:")
        
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Glucose Levels CSV:**
        - Timestamp column
        - Glucose measurement values
        """)
    
    with col2:
        st.markdown("""
        **Food Logs CSV:**
        - Timestamp column
        - Carbohydrates
        - Sugar
        - Calories
        """)
    
    with col3:
        st.markdown("""
        **Accelerometer Data CSV:**
        - Timestamp column
        - Activity measurements
        """)
    
    # Check if data was previously processed
    if st.session_state.data_processed and st.session_state.combined_data is not None:
        combined_data = st.session_state.combined_data
        show_tabs = True
        
elif data_option == "Sample A":
    combined_data_path = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'samples', 'sample_A.csv')
    combined_data = pd.read_csv(combined_data_path)
    st.session_state.combined_data = combined_data
    st.session_state.data_processed = True
    st.success("Sample A loaded successfully!")
    show_tabs = True

elif data_option == "Sample B":
    combined_data_path = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'samples', 'sample_B.csv')
    combined_data = pd.read_csv(combined_data_path)
    st.session_state.combined_data = combined_data
    st.session_state.data_processed = True
    st.success("Sample B loaded successfully!")
    show_tabs = True

# Add some spacing
st.write("")
st.write("")

# Only show tabs if sample data is loaded or user data has been successfully processed
if show_tabs:
    # Create tabs for data exploration
    tab1, tab2, tab3 = st.tabs(["Naive Model", "Machine Learning Model", "Deep Learning Model"])

    with tab1:
        st.subheader("Naive Model")
        
        if st.button('Make prediction', key='naive_button'):
            if combined_data is not None:
                
                # Add your naive model prediction code here
                try:
                    # Call naive model prediction functions
                    column_specs = get_column_specs()
                    prepared_data = prepare_data(combined_data, column_specs["timestamp_column"])
                    
                    train_file = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'train_dataset.csv')
                    train_data = pd.read_csv(train_file)
                    train_data = prepare_data(train_data, column_specs["timestamp_column"])
                    predictions = zeroshot_eval(
                        train_df=train_data,
                        test_df=prepared_data,
                        batch_size=8
                    )
                    
                    # Get all step columns
                    step_columns = [col for col in predictions["predictions_df"].columns if col.startswith("Glucose_step_")]
                    
                    # Apply simple diagonal averaging by patient
                    final_results = simple_diagonal_averaging(
                        predictions["predictions_df"], 
                        prepared_data, 
                        CONTEXT_LENGTH,
                        step_columns
                    )

                    raw_predictions_path = os.path.join(SCRIPT_DIR, '..', 'data', 'outputs', 'sample1.csv')
                    predictions["predictions_df"].to_csv(raw_predictions_path, index=False)
                    
                    # Save final results to CSV
                    final_results_path = os.path.join(SCRIPT_DIR, '..', 'data', 'outputs', 'sample.csv')
                    final_results.to_csv(final_results_path, index=False)
    
                    
                    # Visualize predictions vs actual values
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Filter out zero predictions
                    non_zero_mask = final_results['averaged_prediction'] != 0
                    filtered_results = final_results[non_zero_mask]

                    # Plot predictions (only non-zero values)
                    ax.plot(filtered_results['Timestamp'], filtered_results['averaged_prediction'], 
                            label='Predicted', alpha=0.7)

                    # Plot actual values (all data)
                    ax.plot(final_results['Timestamp'], final_results['Glucose'], 
                            label='Actual', alpha=0.7)

                    ax.set_title('Glucose Predictions vs Actual Values')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Glucose Level')
                    ax.legend()

                    st.pyplot(fig)
                                        
                except Exception as e:
                    st.error(f"Error in naive model prediction: {e}")
            else:
                st.error("Data not available. Please try again.")

    with tab2:
        st.subheader("Machine Learning Model")
        
        if st.button('Make prediction', key='ml_button'):
            if combined_data is not None:
                X_test, y_test = format_dataset(combined_data, CONTEXT_LENGTH, PREDICTION_LENGTH)

                model_output_path = os.path.join(SCRIPT_DIR, '..', 'models', 'xgb_model.pkl')
                joblib.dump(xgb_model, model_output_path)
            
            else:
                st.error("Data not available. Please try again.")

    with tab3:
        st.subheader("Deep Learning Model")
        
        if st.button('Make prediction', key='dl_button'):
            if combined_data is not None:
                st.success("Deep learning model prediction complete!")
                # Add your DL model prediction code here
            else:
                st.error("Data not available. Please try again.")
else:
    st.info("Upload and process data or select a sample dataset to view prediction models.")

# Add some spacing
st.write("")
st.write("")

# App information and disclaimer
st.markdown("""
---
### About this App

This application is designed to help analyze and predict glucose levels based on glucose measurements,
food logs, and physical activity data. The app merges these datasets based on timestamps to identify
patterns and make predictions.

Please note that this is a demonstration tool and should not be used for medical decisions without
consultation with healthcare professionals.

""")

# Add a footer with the custom color
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: #58A618;
    text-align: center;
    padding: 10px;
    border-top: 2px solid #58A618;
}
</style>
<div class="footer">
    <p>Glucose Prediction Application © 2025</p>
</div>
""", unsafe_allow_html=True)