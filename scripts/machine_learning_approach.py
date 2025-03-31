import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Constants for window sizes
X_WINDOW_SIZE = 90 // 5
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
        y_i = X_i[:,:,-y_window_size:,:]
        
        X_i = X_i[:-1]
        y_i = y_i[1:]

        X_list.append(X_i)
        y_list.append(y_i)

    X_matrix = np.concatenate(X_list, axis=0)  
    y_matrix = np.concatenate(y_list, axis=0)

    # Reshaping and cleaning up the matrices
    X_matrix = X_matrix.reshape(X_matrix.shape[0], X_matrix.shape[2], X_matrix.shape[3]) 
    y_matrix = y_matrix.reshape(y_matrix.shape[0], y_matrix.shape[2], y_matrix.shape[3]) 

    # Drop unnecessary columns (timestamp and patient_id)
    X_matrix = X_matrix[:,:,1:-1]   #Chequear si esta guardando index or not
    y_matrix = y_matrix[:,:,-1]    #Checkear si esta seleccionada solo la columna de glucosa

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
def evaluate_model(model, X, y, dataset_name="Validation"):
    """
    Evaluate model performance on the provided dataset.
    """
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    print(f'Mean Absolute Error on {dataset_name} Data: {mae:.4f}')

def plot_results(y_pred, y_true, df):
    """
    
    """

    row_number = 0
    
    for patient in df['patient_id'].unique():
        n_samples = df[df['patient_id'] == patient].shape[0]

        time = np.arange(0, n_samples*5, 5)

        # podria plotear el tramo anterior tambien
        plt.plot(time, y_pred[row_number:row_number+n_samples], label = 'Prediction')
        plt.plot(time, y_true[row_number:row_number+n_samples], label = 'Ground truth')

        plt.xlabel("Time (minutes)")
        plt.ylabel("Blood Glucose Level")
        plt.title(f'Patient{patient}')

        plt.show()


def main():
    print("Running machine_learning_approach script...")

    # Load datasets
    df_train = pd.read_csv('data/processed/train_dataset.csv')
    df_validation = pd.read_csv('data/processed/validation_dataset.csv')
    df_test = pd.read_csv('data/processed/test_dataset.csv')

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

    # Evaluate on the validation set
    evaluate_model(xgb_model, X_val, y_val, "Validation")

    # Re-train on the combined training and validation dataset
    X_train_complete = np.concatenate((X_train, X_val), axis=0)
    y_train_complete = np.concatenate((y_train, y_val), axis=0)
    xgb_model = train_model(xgb_model, X_train_complete, y_train_complete)

    # Evaluate on the test set
    evaluate_model(xgb_model, X_test, y_test, "Test")

    #guardar los resultados del test set

# Main entry point
if __name__ == '__main__':
    main()
