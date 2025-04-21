import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_results(y_pred, y_true, df, sample_number):
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