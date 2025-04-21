import pandas as pd
import os
from sklearn.metrics import root_mean_squared_error
import numpy as np

SEED = 42
TTM_MODEL_PATH = "iaravagni/ttm-finetune-model"
CONTEXT_LENGTH = 52  # 4.33 hrs
PREDICTION_LENGTH = 6  # 30 mins

def main():
    """
    Main function to execute the metrics for the three approaches.
    """
    # Get dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    naive_pred_path = os.path.join(script_dir, '..', 'data', 'outputs', 'naive_predictions.csv')
    ml_pred_path = os.path.join(script_dir, '..', 'data', 'outputs', 'ml_predictions.csv')
    dl_pred_path = os.path.join(script_dir, '..', 'data', 'outputs', 'dl_predictions.csv')

    # Load and prepare data
    naive_pred_df = pd.read_csv(naive_pred_path)
    ml_pred_df = pd.read_csv(ml_pred_path)
    dl_pred_df = pd.read_csv(dl_pred_path)

    y_true = naive_pred_df[['Glucose', 'patient_id']]
    naive_pred = naive_pred_df['averaged_prediction']
    ml_pred = ml_pred_df['averaged_prediction']
    dl_pred = dl_pred_df['averaged_prediction']

    rmse_list = []

    for patient in y_true['patient_id'].unique():
        patient_mask = y_true['patient_id'] == patient

        y_true_patient = y_true[patient_mask].reset_index(drop=True)
        naive_pred_patient = naive_pred[patient_mask].reset_index(drop=True)
        ml_pred_patient = ml_pred[patient_mask].reset_index(drop=True)
        dl_pred_patient = dl_pred[patient_mask].reset_index(drop=True)

        y_true_aux = y_true_patient['Glucose'][CONTEXT_LENGTH:].reset_index(drop=True)
        naive_pred_aux = naive_pred_patient[CONTEXT_LENGTH:].reset_index(drop=True)
        ml_pred_aux = ml_pred_patient[CONTEXT_LENGTH:].reset_index(drop=True)
        dl_pred_aux = dl_pred_patient[CONTEXT_LENGTH:].reset_index(drop=True)

        rmse_naive = np.sqrt(root_mean_squared_error(y_true_aux, naive_pred_aux))
        rmse_ml = np.sqrt(root_mean_squared_error(y_true_aux, ml_pred_aux))
        rmse_dl = np.sqrt(root_mean_squared_error(y_true_aux, dl_pred_aux))

        rmse_list.append([rmse_naive, rmse_ml, rmse_dl])

    model_names = ['Naive', 'ML', 'DL']
    rmse_array = np.array(rmse_list)

    print("Average RMSEs:")
    for name, avg in zip(model_names, rmse_array.mean(axis=0)):
        print(f"  {name}: {avg:.4f}")

    print("\nHighest and Lowest RMSE per model:")
    for i, model in enumerate(model_names):
        max_val = np.max(rmse_array[:, i])
        min_val = np.min(rmse_array[:, i])
        max_patient = y_true['patient_id'].unique()[np.argmax(rmse_array[:, i])]
        min_patient = y_true['patient_id'].unique()[np.argmin(rmse_array[:, i])]
        
        print(f"  {model}:")
        print(f"    Highest RMSE = {max_val:.4f} (Patient {max_patient})")
        print(f"    Lowest RMSE  = {min_val:.4f} (Patient {min_patient})")

    return

if __name__ == "__main__":
    main()

