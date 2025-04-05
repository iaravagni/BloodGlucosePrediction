import time
import json
import timesfm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def load_model():
    # Initialize the TimesFM model with PyTorch backend for CPU
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=1,  # Lower for CPU
            horizon_len=1,  # Prediction steps (30 min at 5-min intervals)
            num_layers=50,
            use_positional_embedding=False,
            context_len=2048,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
    )


def main():
    print("Running machine_learning_approach script...")

    # Load datasets
    df_train = pd.read_csv('data/processed/train_dataset.csv')
    df_validation = pd.read_csv('data/processed/validation_dataset.csv')
    df_test = pd.read_csv('data/processed/test_dataset.csv')

    
    # Load input data
    with open("../data/processed/data.json", "r") as f:
        input_data = json.load(f)

    metrics = {"eval_mae_xreg_timesfm": [], "eval_mse_xreg_timesfm": []}

    for i, example in enumerate(input_data.values()):
        start_time = time.time()
        
        # Forecast with covariates
        cov_forecast, _ = model.forecast_with_covariates(  
            inputs=example["glucose_inputs"],
            dynamic_numerical_covariates={"accelerometer": example["accelerometer"],"calories": example["calories"], "sugar": example["sugar"], "carbs": example["carbs"],},
            dynamic_categorical_covariates={"time_window": example["time_window"]},
            static_numerical_covariates={"HbA1c": example["HbA1c"]},
            static_categorical_covariates={"gender": example["gender"]},
            freq=[5] * len(example["glucose_inputs"]),
            xreg_mode="xreg + timesfm",  # Default mode
            ridge=0.0,
            force_on_cpu=True,
            normalize_xreg_target_per_input=True,  # Default
        )
        
        print(f"\rFinished batch {i} in {time.time() - start_time:.2f} seconds", end="")
        
        metrics["eval_mae_xreg_timesfm"].append(mae(cov_forecast, example["glucose outputs"]))
        metrics["eval_mse_xreg_timesfm"].append(mse(cov_forecast, example["glucose outputs"]))

    print("\n")

    for k, v in metrics.items():
        print(f"{k}: {np.mean(v)}")