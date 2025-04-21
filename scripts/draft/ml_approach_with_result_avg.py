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
        y_i = np.lib.stride_tricks.sliding_window_view(df_i.values, (y_window_size, df_i.shape[1]))

        # y_i = X_i[:,:,-y_window_size:,:]
        
        # X_i = X_i[:-1]
        # y_i = y_i[1:]
        X_i = X_i[:-y_window_size]  # Remove last few X samples that wouldn't have a full y
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
def evaluate_model(model, y_true, y_pred, dataset_name="Validation"):
    """
    Evaluate model performance on the provided dataset.
    """    
    mae = mean_absolute_error(y_true, y_pred)
    print(f'Mean Absolute Error on {dataset_name} Data: {mae:.4f}')

def avg_result(y):

    print(y.shape)

    return y


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

    y_val_pred = xgb_model.predict(X_val)

    print(y_val_pred.shape)
    print(y_val_pred)

    # print(y_val.shape)
    # print(y_val)

    # y_val_pred[0] = y_val_pred[0,0]
    
    # y_val_pred[1] = y_val_pred[1,0] + y_val_pred[0,1]

    # y_val_pred[2] = y_val_pred[2,0] + y_val_pred[1,1] + y_val_pred[0,2]

    y_val_pred_resized = np.zeros([y_val_pred.shape[0],1])

    for i in range(y_val_pred.shape[0]):
        j = 0
        k = i
        while k>=0 and j<6:
            y_val_pred_resized[i,0] += y_val_pred[k,j]

            if i == 0:
                print(y_val_pred[k,j])
            
            k -= 1
            j += 1
            

        y_val_pred_resized[i,0] = y_val_pred_resized[i,0] / (j)

    print(y_val_pred_resized.shape)
    print(y_val_pred_resized)

    y_val_true = df_validation[['Glucose', 'patient_id']]

    y_val_true = []

    for patient in df_validation['patient_id'].unique():

        total_samples = (df_validation[df_validation['patient_id'] == patient]).shape[0] - X_WINDOW_SIZE

        y_val_true_i = df_validation[['Glucose', 'patient_id']][df_validation['patient_id'] == patient].iloc[-total_samples:, :]

        y_val_true.append(y_val_true_i)
    
    y_val_true = np.concatenate(y_val_true, axis=0)[:,0]

    print(y_val_true.shape)
    print(y_val_true)

    

    

    # # y_pred = avg_result(y_pred)

    # # Evaluate on the validation set
    # evaluate_model(xgb_model, X_val, y_val, "Validation")

    # # Re-train on the combined training and validation dataset
    # X_train_complete = np.concatenate((X_train, X_val), axis=0)
    # y_train_complete = np.concatenate((y_train, y_val), axis=0)
    # xgb_model = train_model(xgb_model, X_train_complete, y_train_complete)

    # # Evaluate on the test set
    # evaluate_model(xgb_model, X_test, y_test, "Test")

    # #guardar los resultados del test set

# Main entry point
if __name__ == '__main__':
    main()
