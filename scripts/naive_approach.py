import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timesfm
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Constants for window sizes
X_WINDOW_SIZE = int(6 * 60 // 5)  # 6 hours in 5-minute intervals
Y_WINDOW_SIZE = int(0.5 * 60 // 5)  # 30 minutes in 5-minute intervals

def get_batched_data_fn(
    df,
    batch_size=128, 
    context_len=X_WINDOW_SIZE, 
    horizon_len=Y_WINDOW_SIZE,
):
    """
    Create batched data from the dataframe for model training/inference.
    
    Args:
        df: DataFrame containing patient data
        batch_size: Number of examples per batch
        context_len: Length of input context window
        horizon_len: Length of prediction horizon
        
    Returns:
        Function that yields batches of data
    """
    examples = defaultdict(list)
    num_examples = 0
    
    for patient in df['patient_id'].unique():
        sub_df = df[df["patient_id"] == patient] 
        for start in range(0, len(sub_df) - (int(context_len) + int(horizon_len)), int(horizon_len)):
            num_examples += 1
            context_end = start + context_len
            
            examples["patient_id"].append(patient)
            examples["gender"].append(sub_df.iloc[0]["Gender"])
            examples["HbA1c"].append(sub_df.iloc[0]['HbA1c'])
            examples["inputs"].append(sub_df["Glucose"][start:context_end].tolist())
            examples["accelerometer"].append(sub_df["Accelerometer"][start:context_end + horizon_len].tolist())
            examples["calories"].append(sub_df["Calories"][start:context_end + horizon_len].tolist())
            examples["sugar"].append(sub_df["Sugar"][start:context_end + horizon_len].tolist())
            examples["carbs"].append(sub_df["Carbs"][start:context_end + horizon_len].tolist())
            examples["timestamp"].append(sub_df["Timestamp"][start:context_end + horizon_len].tolist())
            examples["outputs"].append(sub_df["Glucose"][context_end:(context_end + horizon_len)].tolist())
  
    def data_fn():
        for i in range(1 + (num_examples - 1) // batch_size):
            yield {k: v[(i * batch_size):((i + 1) * batch_size)] for k, v in examples.items()}
  
    return data_fn

def save_input_data(input_data, filename="input_data.json"):
    """
    Save the batched input data to a JSON file.
    
    Args:
        input_data: Function that yields batches of data
        filename: Path to save the JSON file
    """
    batched_data = list(input_data())  # Convert generator to list
    
    # Convert NumPy types to Python native types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Apply conversion to every element
    batched_data = json.loads(json.dumps(batched_data, default=convert_types))

    with open(filename, "w") as f:
        json.dump(batched_data, f, indent=4)

    print(f"Input data saved to {filename}")

def load_model():
    """
    Load the TimesFM model for forecasting.
    
    Returns:
        TimesFM model instance
    """
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=32,
            horizon_len=128,
            context_len=512,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )
    return tfm

def run_inference(model, input_data):
    """
    Run inference on the given input data using the TimesFM model.
    
    Args:
        model: TimesFM model instance
        input_data: Function that yields batches of data
        
    Returns:
        Tuple of (all_forecasts, metrics)
    """
    # Initialize metrics dictionary for tracking performance
    metrics = defaultdict(list)
    
    # Initialize dictionary to store all forecasts
    all_forecasts = {
        "patient_id": [],
        "timestamp": [],
        "raw_forecast": [],
        "cov_forecast": [],
        "ols_forecast": [],
        "ground_truth": []
    }
    
    # Process each batch
    for i, example in enumerate(input_data()):
        start_time = time.time()
        
        # Generate forecast without covariates
        raw_forecast, _ = model.forecast(
            inputs=example["inputs"], freq=[0] * len(example["inputs"])
        )
        
        # Generate forecast with covariates
        cov_forecast, ols_forecast = model.forecast_with_covariates(  
            inputs=example["inputs"],
            dynamic_numerical_covariates={
                "accelerometer": example["accelerometer"],
                "calories": example["calories"],
                "sugar": example["sugar"],
                "carbs": example["carbs"],
            },
            dynamic_categorical_covariates={},
            static_numerical_covariates={},
            static_categorical_covariates={
                "gender": example["gender"],
                "HbA1c": example["HbA1c"],
            },
            freq=[0] * len(example["inputs"]),
            xreg_mode="xreg + timesfm",  # default
            ridge=0.0,
            force_on_cpu=False,
            normalize_xreg_target_per_input=True,  # default
        )
        
        elapsed_time = time.time() - start_time
        print(f"\rBatch {i+1}: processed in {elapsed_time:.2f} seconds", end="")
        
        # Store all forecasts for CSV output
        for j in range(len(example["outputs"])):
            all_forecasts["patient_id"].append(example["patient_id"][j])
            all_forecasts["timestamp"].append(example["timestamp"][j][-Y_WINDOW_SIZE:])
            all_forecasts["raw_forecast"].append(raw_forecast[j, :Y_WINDOW_SIZE].tolist())
            all_forecasts["cov_forecast"].append(cov_forecast[j].tolist())
            all_forecasts["ols_forecast"].append(ols_forecast[j].tolist())
            all_forecasts["ground_truth"].append(example["outputs"][j])

        # Calculate metrics for monitoring
        metrics["eval_mae_timesfm"].extend(
            mae(raw_forecast[:, :Y_WINDOW_SIZE], example["outputs"])
        )
        metrics["eval_mae_xreg_timesfm"].extend(
            mae(cov_forecast, example["outputs"])
        )
        metrics["eval_mae_xreg"].extend(
            mae(ols_forecast, example["outputs"])
        )
        metrics["eval_mse_timesfm"].extend(
            mse(raw_forecast[:, :Y_WINDOW_SIZE], example["outputs"])
        )
        metrics["eval_mse_xreg_timesfm"].extend(
            mse(cov_forecast, example["outputs"])
        )
        metrics["eval_mse_xreg"].extend(
            mse(ols_forecast, example["outputs"])
        )
    
    print("\nInference complete!")
    return all_forecasts, metrics

def mse(y_pred, y_true):
    """
    Calculate Mean Squared Error.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        MSE values per sample
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.mean(np.square(y_pred - y_true), axis=1, keepdims=True)

def mae(y_pred, y_true):
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        MAE values per sample
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.mean(np.abs(y_pred - y_true), axis=1, keepdims=True)

def save_forecasts_to_csv(forecasts_data, output_file="forecasts.csv"):
    """
    Save model forecasts to a CSV file.
    
    Args:
        forecasts_data: Dictionary containing forecast data
        output_file: Path to save the CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create a list to store all forecast rows
    forecast_rows = []
    
    # Process each example
    for i in range(len(forecasts_data["patient_id"])):
        patient_id = forecasts_data["patient_id"][i]
        timestamps = forecasts_data["timestamp"][i]
        
        # Get forecasts for this example
        raw_forecast = forecasts_data["raw_forecast"][i]
        cov_forecast = forecasts_data["cov_forecast"][i]
        ols_forecast = forecasts_data["ols_forecast"][i]
        ground_truth = forecasts_data["ground_truth"][i]
        
        # Create rows for each timestamp in the forecast horizon
        for j in range(len(ground_truth)):
            forecast_rows.append({
                "patient_id": patient_id,
                "timestamp": timestamps[j],
                "raw_forecast": raw_forecast[j],
                "cov_forecast": cov_forecast[j],
                "ols_forecast": ols_forecast[j],
                "ground_truth": ground_truth[j]
            })
    
    # Convert to DataFrame and save as CSV
    forecasts_df = pd.DataFrame(forecast_rows)
    forecasts_df.to_csv(output_file, index=False)
    
    print(f"Forecasts saved to {output_file}")

def main():
    """Main function to run the glucose forecasting pipeline."""
    print("Running glucose forecasting pipeline...")

    # Load test dataset
    test_file = "data/processed/test_dataset.csv"
    print(f"Loading test data from {test_file}")
    df_test = pd.read_csv(test_file)

    # Get batched data function
    batch_size = 128
    print(f"Creating batched data with batch size {batch_size}")
    input_data = get_batched_data_fn(df_test, batch_size=batch_size)

    # Load the model
    print("Loading TimesFM model...")
    model = load_model()

    # Run inference
    print("Running inference...")
    all_forecasts, metrics = run_inference(model, input_data)

    # Save forecasts to CSV
    print("Saving forecasts to CSV...")
    save_forecasts_to_csv(all_forecasts, output_file="results/glucose_forecasting/forecasts.csv")
    
    # Print results
    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}
    print("\nAverage Metrics: ")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()