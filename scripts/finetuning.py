import os
import tempfile
import pandas as pd
import warnings
import torch
import numpy as np
import math
from transformers import Trainer, TrainingArguments, set_seed, EarlyStoppingCallback, Trainer
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from tsfm_public import TimeSeriesPreprocessor
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.callbacks import TrackingCallback

from huggingface_hub import login, create_repo, upload_folder


SEED = 42
TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
CONTEXT_LENGTH = 52  # 4.33 hrs
PREDICTION_LENGTH = 6  # 30 mins

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

def create_dataset(ts_preprocessor, dataframe, train_df, column_specs, context_length, prediction_length, stride=1):
    """
    Create a ForecastDFDataset using the proper parameters based on the example.
    
    Args:
        dataframe: Pandas dataframe with time series data
        column_specs: Dictionary with column specifications
        context_length: Context window length
        prediction_length: Prediction horizon length
        stride: Stride for sliding window
        
    Returns:
        ForecastDFDataset instance
    """

    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(dataframe[column_specs["timestamp_column"]]):
        dataframe[column_specs["timestamp_column"]] = pd.to_datetime(dataframe[column_specs["timestamp_column"]])

    # Standardize the test dataframe
    dataframe = ts_preprocessor._standardize_dataframe(dataframe)

    ts_preprocessor.train(train_df)
    
   
    # Preprocess the test data
    dataframe_prep = dataframe.copy()  # Skip preprocessing to avoid scaling errors
    
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
    params["enable_padding"] = True
    
    # Add frequency token - this is critical for TinyTimeMixer
    params["frequency_token"] = ts_preprocessor.get_frequency_token(ts_preprocessor.freq)
    
    # Create the ForecastDFDataset
    dataset = ForecastDFDataset(dataframe_prep, **params)
  

    return dataset


def finetune(train_df, valid_df, learning_rate,num_epochs,batch_size, OUT_DIR, context_length=CONTEXT_LENGTH, forecast_length=PREDICTION_LENGTH):
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(OUT_DIR, "logs"),  # Specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.001,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    column_specifiers = get_column_specs()

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

    # Create datasets
    print("Creating training dataset...")
    train_dataset = create_dataset(
        tsp,
        dataframe=train_df,
        train_df=train_df,
        column_specs=column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length
    )
    
    print("Creating validation dataset...")
    valid_dataset = create_dataset(
        tsp,
        dataframe=valid_df,
        train_df=train_df,
        column_specs=column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length
    )

    finetune_forecast_model = get_model(
        TTM_MODEL_PATH,
        context_length=context_length,
        prediction_length=forecast_length,
        num_input_channels=tsp.num_input_channels,
        decoder_mode="mix_channel",  # ch_mix:  set to mix_channel for mixing channels in history
        prediction_channel_indices=tsp.prediction_channel_indices,
    )

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(train_dataset) / (batch_size)),
    )

    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )

    # Fine tune
    finetune_forecast_trainer.train()

    return finetune_forecast_trainer.model

def upload_to_hf(model):
    model.save_pretrained("model/finetuned_ttm_model")

    username = 'iaravagni'
    repo_name = "ttm-finetune-model"  # customize this

    upload_folder(
        repo_id=f"{username}/{repo_name}",
        folder_path="./model",  # path to your trained model dir
        path_in_repo="",  # root of the repo
    )

    return


def main():

    # Get dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    validation_file = os.path.join(script_dir, '..', 'data', 'processed', 'validation_dataset.csv')
    train_file = os.path.join(script_dir, '..', 'data', 'processed', 'train_dataset.csv')
    
    # Load and prepare data
    validation_data = pd.read_csv(validation_file)
    train_data = pd.read_csv(train_file)

    # Load and prepare data
    validation_data = pd.read_csv("/content/validation_dataset.csv")
    train_data = pd.read_csv("/content/train_dataset.csv")


    learning_rate = 0.001
    num_epochs = 40
    batch_size = 32

    OUT_DIR = "model"

    ttm_finetuned_model = finetune(train_data, validation_data, learning_rate,num_epochs,batch_size, OUT_DIR)

    return


# Main entry point
if __name__ == '__main__':
    main()
