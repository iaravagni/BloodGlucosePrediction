import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import joblib


# Constants for window sizes
X_WINDOW_SIZE =  52
Y_WINDOW_SIZE = 30 // 5

def format_dataset(df, X_window_size, y_window_size):
    """
    Format the dataset by applying sliding window technique to the dataframe and prepare the input features and labels.
    """
    X_list, y_list = [], []
    
    for patient in df['patient_id'].unique():
        df_i = df[df['patient_id'] == patient]
        
        # Sliding window view to generate features and labels
        X_i = np.lib.stride_tricks.sliding_window_view(df_i.values, (X_window_size, df_i.shape[1]))
        y_i = np.lib.stride_tricks.sliding_window_view(df_i.values, (y_window_size, df_i.shape[1]))

        X_i = X_i[:-y_window_size]
        y_i = y_i[X_window_size:]
        
        X_list.append(X_i)
        y_list.append(y_i)

    X_matrix = np.concatenate(X_list, axis=0)  
    y_matrix = np.concatenate(y_list, axis=0)

    # Reshaping and cleaning up the matrices
    X_matrix = X_matrix.reshape(X_matrix.shape[0], X_matrix.shape[2], X_matrix.shape[3]) 
    y_matrix = y_matrix.reshape(y_matrix.shape[0], y_matrix.shape[2], y_matrix.shape[3]) 

    # Drop unnecessary columns (timestamp and patient_id)
    X_matrix = X_matrix[:,:,2:-1]
    y_matrix = y_matrix[:,:,2]

    # Flatten X and y for XGBoost input
    X_flat = X_matrix.reshape(X_matrix.shape[0], -1)
    y_flat = y_matrix.reshape(y_matrix.shape[0], -1)  

    return X_flat, y_flat

# Function to train the model
def train_model(model, X_train, y_train):
    """
    Train the given model with the training data.
    """
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(y_true, y_pred, dataset_name="Validation"):
    """
    Evaluate model performance on the provided dataset.
    """    
    rmse = root_mean_squared_error(y_true, y_pred)
    print(f'Root Mean Squared Error on {dataset_name} Data: {rmse:.4f}')

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
    print("Running machine_learning_approach script...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, '..', 'data', 'processed', 'test_dataset.csv')
    train_file = os.path.join(script_dir, '..', 'data', 'processed', 'train_dataset.csv')
    validation_file = os.path.join(script_dir, '..', 'data', 'processed', 'validation_dataset.csv')

    # Load datasets
    df_train = pd.read_csv(train_file)
    df_validation = pd.read_csv(validation_file)
    df_test = pd.read_csv(test_file)

    # Format datasets
    X_train, y_train = format_dataset(df_train, X_WINDOW_SIZE, Y_WINDOW_SIZE)
    X_val, y_val = format_dataset(df_validation, X_WINDOW_SIZE, Y_WINDOW_SIZE)
    X_test, y_test = format_dataset(df_test, X_WINDOW_SIZE, Y_WINDOW_SIZE)

    # Initialize the model
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        objective='reg:squarederror', 
        random_state=42
    )

    # Train model on the training dataset
    xgb_model = train_model(xgb_model, X_train, y_train)

    model_output_path = os.path.join(script_dir, '..', 'data', 'outputs', 'xgb_model.pkl')
    joblib.dump(xgb_model, model_output_path)

    xgb_model = joblib.load(model_output_path)

    y_val_pred = xgb_model.predict(X_val)

    # Evaluate on the validation set
    evaluate_model(y_val, y_val_pred, "Validation")

    # Re-train on the combined training and validation dataset
    X_train_complete = np.concatenate((X_train, X_val), axis=0)
    y_train_complete = np.concatenate((y_train, y_val), axis=0)
    xgb_model = train_model(xgb_model, X_train_complete, y_train_complete)

    y_test_pred = xgb_model.predict(X_test)

    # Evaluate on the test set
    evaluate_model(y_test, y_test_pred, "Test")

    output_dir = os.path.join(script_dir, '..', 'data', 'outputs', 'ml_predictions_raw.csv')

    # Save test set results
    pd.DataFrame(y_test_pred).to_csv(output_dir)

    step_columns = [f"step_{i}" for i in range(y_test_pred.shape[1])]


    final_results = simple_diagonal_averaging(
        pd.DataFrame(y_test_pred), 
        df_test, 
        X_WINDOW_SIZE,
        pd.DataFrame(y_test_pred).columns
    )
    
    # Save final results to CSV
    final_results_path = os.path.join(script_dir, '..', 'data', 'outputs', 'ml_predictions.csv')
    final_results.to_csv(final_results_path, index=False)
    
    return



# Main entry point
if __name__ == '__main__':
    main()
