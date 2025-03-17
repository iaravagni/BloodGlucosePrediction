import timesfm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# Initialize the TimesFM model with PyTorch backend for CPU
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu",
        per_core_batch_size=1,  # Lower for CPU
        horizon_len=6,  # Set to prediction steps (30 min at 5-min intervals)
        num_layers=50,
        use_positional_embedding=False,
        context_len=2048,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
)

# Data directory containing the patient files
data_dir = "../data/processed/dataset_by_patient"  # Update this path
file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Metrics storage
all_metrics = []

# Process each patient file
for file_name in file_list:
    print(f"Processing {file_name}")
    file_path = os.path.join(data_dir, file_name)
    
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Rename columns to match TimesFM expected format
    df = df.rename(columns={'Timestamp': 'ds', 'Glucose (t-0)': 'y'})
    
    # Add unique_id column (patient identifier)
    patient_id = file_name.split('.')[0]
    df['unique_id'] = patient_id
    
    # Keep only required columns for TimesFM
    input_df = df[['unique_id', 'ds', 'y']]
    
    # Handle missing values
    # input_df = input_df.ffill().bfill().fillna(0)
    
    # Determine frequency from data
    # Assuming timestamps are 5 minutes apart
    freq = pd.infer_freq(input_df['ds'])
    if freq is None:
        # If can't infer, default to 5-minute frequency
        freq = '5min'
    
    try:
        # Generate forecasts
        forecast_df = tfm.forecast_on_df(
            inputs=input_df,
            freq=freq,
            value_name="y",
            num_jobs=1,
        )

        # Identify the correct prediction column
        predicted_col = "timesfm"  # Main forecast output

        # Rename for consistency
        forecast_df = forecast_df.rename(columns={predicted_col: "y_pred"})

        print(f"Forecast DataFrame:\n{forecast_df.head()}\n")
        print(f"Input DataFrame:\n{input_df.head()}\n")

        print(f"Forecast DataFrame Timestamp dtype: {forecast_df['ds'].dtype}")
        print(f"Input DataFrame Timestamp dtype: {input_df['ds'].dtype}")



        # Merge forecasts with actuals
        merged_df = forecast_df.merge(
            input_df, 
            on=['unique_id', 'ds'], 
            how='left'
        )

        # Ensure column exists before evaluation
        if "y_pred" not in merged_df.columns:
            raise ValueError(f"Expected prediction column 'y_pred' not found after merging")
        # Rename to standard name for consistency
        merged_df = merged_df.rename(columns={predicted_col: 'y_pred'})

        # Filter rows where we have both predictions and actuals
        eval_df = merged_df.dropna(subset=['y', 'y_pred'])
        
        # Calculate metrics
        mae = mean_absolute_error(eval_df['y'], eval_df['y_pred'])
        rmse = math.sqrt(mean_squared_error(eval_df['y'], eval_df['y_pred']))
        r2 = r2_score(eval_df['y'], eval_df['y_pred'])
        mard = np.mean(np.abs((eval_df['y_pred'] - eval_df['y']) / eval_df['y'])) * 100
        
        # Store metrics
        patient_metrics = {
            'patient': patient_id,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mard': mard
        }
        all_metrics.append(patient_metrics)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        # Plot actuals
        plt.plot(eval_df['ds'], eval_df['y'], label='Actual Glucose')
        
        # Plot predictions
        plt.plot(eval_df['ds'], eval_df['y_pred'], 'r--', label='Predicted Glucose')
        
        plt.title(f'Glucose Prediction for Patient {patient_id}')
        plt.xlabel('Time')
        plt.ylabel('Glucose Level (mg/dL)')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(f'prediction_{patient_id}.png')
        plt.close()
        
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        continue

# Create a DataFrame with all patient metrics
metrics_df = pd.DataFrame(all_metrics)

# Calculate average metrics across all patients
if not metrics_df.empty:
    avg_metrics = {}
    for col in metrics_df.columns:
        if col != 'patient':
            avg_metrics[col] = metrics_df[col].mean()
    
    # Add average metrics to the DataFrame
    avg_metrics['patient'] = 'AVERAGE'
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_metrics])], ignore_index=True)
    
    # Save metrics to CSV
    metrics_df.to_csv('glucose_prediction_metrics.csv', index=False)
    
    print("All patients processed successfully.")
    print("Metrics saved to glucose_prediction_metrics.csv")
else:
    print("No metrics collected. Check for errors in processing.")