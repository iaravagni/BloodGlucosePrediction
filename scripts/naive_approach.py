import os
import tempfile
import pandas as pd
import warnings
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, set_seed

from tsfm_public import TimeSeriesPreprocessor
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.dataset import ForecastDFDataset

# Constants
SEED = 42
TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
CONTEXT_LENGTH = 52  # 4.33 hrs
PREDICTION_LENGTH = 6  # 30 mins
OUT_DIR = "ttm_finetuned_models/"

def setup_environment():
    """
    Set up the environment for model training and evaluation.
    Creates necessary directories and sets random seed for reproducibility.
    """
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'results'), exist_ok=True)
    
def load_dataset(file_path):
    """
    Load the dataset from the specified file path.
    
    Args:
        file_path (str): Path to the dataset CSV file
        
    Returns:
        pd.DataFrame: The loaded dataset
    """
    return pd.read_csv(file_path)

def prepare_data(data, timestamp_column):
    """
    Prepare the dataset by converting timestamp column to datetime format.
    
    Args:
        data (pd.DataFrame): The dataset
        timestamp_column (str): Name of the timestamp column
        
    Returns:
        pd.DataFrame: The processed dataset
    """
    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    return data

def get_column_specs():
    """
    Define and return column specifications for the dataset.
    
    Returns:
        dict: Column specifications including timestamp, ID, target, and control columns
    """
    timestamp_column = "Timestamp"
    id_columns = ["patient_id"]
    target_columns = ["Glucose"]
    control_columns = ["Accelerometer", "Calories", "Carbs", "Sugar", "Gender", "HbA1c", "Age"]
    
    return {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": target_columns,
        "control_columns": control_columns,
    }

def create_test_only_dataset(ts_preprocessor, test_dataset, train_dataset=None, stride=1, enable_padding=True, **dataset_kwargs):
    """
    Creates a preprocessed pytorch dataset for testing only.
    
    Args:
        ts_preprocessor: TimeSeriesPreprocessor instance
        test_dataset: Pandas dataframe for testing
        train_dataset: Optional pandas dataframe for training the scaler
        stride: Stride used for creating the dataset
        enable_padding: If True, datasets are created with padding
        dataset_kwargs: Additional keyword arguments to pass to ForecastDFDataset
    
    Returns:
        ForecastDFDataset for testing
    """
    # Standardize the test dataframe
    test_data = ts_preprocessor._standardize_dataframe(test_dataset)
    
    # Train the preprocessor on the training data if provided, otherwise use test data
    if train_dataset is not None:
        train_data = ts_preprocessor._standardize_dataframe(train_dataset)
        ts_preprocessor.train(train_data)
    else:
        ts_preprocessor.train(test_data)
    
    # Preprocess the test data
    test_data_prep = test_data.copy()  # Skip preprocessing to avoid scaling errors
    
    # Specify columns
    column_specifiers = {
        "id_columns": ts_preprocessor.id_columns,
        "timestamp_column": ts_preprocessor.timestamp_column,
        "target_columns": ts_preprocessor.target_columns,
        "observable_columns": ts_preprocessor.observable_columns,
        "control_columns": ts_preprocessor.control_columns,
        "conditional_columns": ts_preprocessor.conditional_columns,
        "categorical_columns": ts_preprocessor.categorical_columns,
        "static_categorical_columns": ts_preprocessor.static_categorical_columns,
    }
    
    params = column_specifiers
    params["context_length"] = ts_preprocessor.context_length
    params["prediction_length"] = ts_preprocessor.prediction_length
    params["stride"] = stride
    params["enable_padding"] = enable_padding
    
    # Add frequency token - this is critical for TinyTimeMixer
    params["frequency_token"] = ts_preprocessor.get_frequency_token(ts_preprocessor.freq)
    
    # Update with any additional kwargs
    params.update(**dataset_kwargs)
    
    # Create the ForecastDFDataset
    test_dataset = ForecastDFDataset(test_data_prep, **params)
    
    if len(test_dataset) == 0:
        raise RuntimeError("The generated test dataset is of zero length.")
    
    return test_dataset

    

def zeroshot_eval(train_df, test_df, batch_size, context_length=CONTEXT_LENGTH, forecast_length=PREDICTION_LENGTH):
    """
    Performs zero-shot evaluation of time series forecasting on test data.
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe
        batch_size: Batch size for evaluation
        context_length: Number of time steps to use as context
        forecast_length: Number of time steps to predict
        
    Returns:
        dict: Dictionary containing predictions dataframe and metrics
    """
    column_specifiers = get_column_specs()
    
    # Create preprocessor with scaling disabled
    tsp = TimeSeriesPreprocessor(
        timestamp_column=column_specifiers["timestamp_column"],
        id_columns=column_specifiers["id_columns"],
        target_columns=column_specifiers["target_columns"],
        control_columns=column_specifiers["control_columns"],
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=False,
        encode_categorical=False,
        force_return="zeropad",
    )
    
    # Load model
    zeroshot_model = get_model(
        TTM_MODEL_PATH,
        context_length=context_length,
        prediction_length=forecast_length,
        freq_prefix_tuning=False,
        freq=None,
        prefer_l1_loss=False,
        prefer_longer_context=True,
    )
    
    # Create test dataset
    dset_test = create_test_only_dataset(ts_preprocessor=tsp, test_dataset=test_df, train_dataset=train_df)
    
    # Setup trainer
    temp_dir = tempfile.mkdtemp()
    zeroshot_trainer = Trainer(
        model=zeroshot_model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=batch_size,
            seed=SEED,
            report_to="none",
        ),
    )
    
    # Get predictions
    predictions_dict = zeroshot_trainer.predict(dset_test)
    
    # Process predictions
    processed_predictions = process_predictions(predictions_dict, tsp, column_specifiers["target_columns"])
    
    # Get evaluation metrics
    metrics = zeroshot_trainer.evaluate(dset_test)
    
    return {
        "predictions_df": processed_predictions,
        "metrics": metrics
    }

def process_predictions(predictions_dict, tsp, target_columns):
    """
    Process the predictions from the Trainer into a usable DataFrame.
    
    Args:
        predictions_dict: Predictions from the Trainer
        tsp: TimeSeriesPreprocessor instance
        target_columns: List of target column names
        
    Returns:
        pd.DataFrame: DataFrame containing processed predictions
    """
    # Extract predictions
    if hasattr(predictions_dict, 'predictions'):
        raw_predictions = predictions_dict.predictions
    else:
        raw_predictions = predictions_dict.get('predictions', predictions_dict)
    
    # Handle tuple predictions (mean and uncertainty)
    if isinstance(raw_predictions, tuple):
        predictions = raw_predictions[0]
    else:
        predictions = raw_predictions
    
    # Get shape information
    n_samples, n_timesteps, n_features = predictions.shape
    
    # Create DataFrame for processed predictions
    processed_df = pd.DataFrame()
    
    # Extract predictions for each target and timestep
    for i, col in enumerate(target_columns):
        if i < n_features:
            for t in range(n_timesteps):
                processed_df[f"{col}_step_{t+1}"] = predictions[:, t, i]
    
    return processed_df

def simple_diagonal_averaging(predictions_df, test_data, context_length, step_columns):
    """
    Simple approach to diagonally averaging predictions by patient.
    Skips the first context_length rows and averages the rest for each timestamp.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with step-wise predictions
        test_data (pd.DataFrame): Original test data with patient IDs
        context_length (int): Number of context steps used in the model
        step_columns (list): List of step column names
    
    Returns:
        pd.DataFrame: DataFrame with averaged predictions
    """
    # Create a new dataframe for the final results
    final_df = test_data.copy()
    
    # Initialize prediction column with zeros/NaN
    final_df['averaged_prediction'] = 0
    
    # Process each patient separately
    for patient_id in test_data['patient_id'].unique():
        # Get indices for this patient
        patient_mask = final_df['patient_id'] == patient_id
        patient_indices = final_df[patient_mask].index
        
        # Skip the first context_length rows for this patient
        start_idx = min(context_length, len(patient_indices))
        
        # For each row after the context window
        for i in range(start_idx, len(patient_indices)):
            row_idx = patient_indices[i]
            pred_row_idx = i - context_length
            
            # Skip if the prediction row index is negative
            if pred_row_idx < 0:
                continue
                
            # Get the corresponding prediction row
            if pred_row_idx < len(predictions_df):
                # Average the predictions for all steps
                avg_prediction = predictions_df.iloc[pred_row_idx][step_columns].mean()
                final_df.loc[row_idx, 'averaged_prediction'] = avg_prediction
    
    return final_df

def main():
    """
    Main function to execute the time series forecasting workflow.
    """
    # Setup
    # setup_environment()
    
    # Get dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, '..', 'data', 'processed', 'test_dataset.csv')
    train_file = os.path.join(script_dir, '..', 'data', 'processed', 'train_dataset.csv')
    
    # Load and prepare data
    test_data = load_dataset(test_file)
    train_data = load_dataset(train_file)
    column_specs = get_column_specs()
    test_data = prepare_data(test_data, column_specs["timestamp_column"])
    train_data = prepare_data(train_data, column_specs["timestamp_column"])
    
    # Run zero-shot evaluation
    results = zeroshot_eval(
        train_df=train_data,
        test_df=test_data,
        batch_size=8,
        context_length=CONTEXT_LENGTH,
        forecast_length=PREDICTION_LENGTH
    )

    # Get all step columns
    step_columns = [col for col in results["predictions_df"].columns if col.startswith("Glucose_step_")]
    
    # Apply simple diagonal averaging by patient
    final_results = simple_diagonal_averaging(
        results["predictions_df"], 
        test_data, 
        CONTEXT_LENGTH,
        step_columns
    )
    
    # Save raw predictions to CSV
    raw_predictions_path = os.path.join(script_dir, '..', 'data', 'outputs', 'naive_predictions_raw.csv')
    results["predictions_df"].to_csv(raw_predictions_path, index=False)
    
    # Save final results to CSV
    final_results_path = os.path.join(script_dir, '..', 'data', 'outputs', 'naive_predictions.csv')
    final_results.to_csv(final_results_path, index=False)
    
    return

if __name__ == "__main__":
    main()