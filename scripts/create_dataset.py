import pandas as pd
import numpy as np
import json


def create_dataframes():
    data_path = f"../data/raw/big_ideas_dataset"

    for i in range(1,17):
        if i<10:
            patient = "00"+str(i)
        else:
            patient = "0"+str(i)

        print("Patient"+str(i))
                
        # Load files
        bg_df = pd.read_csv(f"{data_path}/{patient}/Dexcom_{patient}.csv")
        hr_df = pd.read_csv(f"{data_path}/{patient}/HR_{patient}.csv")
        acc_df = pd.read_csv(f"{data_path}/{patient}/ACC_{patient}.csv")
        food_df = pd.read_csv(f"{data_path}/{patient}/Food_Log_{patient}.csv")
        demographic_data = pd.read_csv(f"{data_path}/Demographics.csv")

        # Clean and convert 'Timestamp' columns to datetime format
        bg_df['Timestamp'] = pd.to_datetime(bg_df['Timestamp (YYYY-MM-DDThh:mm:ss)'], errors='coerce')
        hr_df['datetime'] = pd.to_datetime(hr_df['datetime'], errors='coerce')
        acc_df['datetime'] = pd.to_datetime(acc_df['datetime'], errors='coerce')
        food_df['time_begin'] = pd.to_datetime(food_df['time_begin'], errors='coerce')
        
        # Set the Timestamp as index for all dataframes
        bg_df.set_index('Timestamp', inplace=True)
        hr_df.set_index('datetime', inplace=True)
        acc_df.set_index('datetime', inplace=True)
        # food_df.set_index('time_begin', inplace=True)

        # Filter the rows where Event Type is 'EGV'
        bg_df = bg_df[bg_df['Event Type'] == 'EGV']

        # Calculate magnitude for accelerometer data
        acc_df['Magnitude'] = np.sqrt(acc_df[' acc_x']**2 + acc_df[' acc_y']**2 + acc_df[' acc_z']**2).round(2)
        acc_df['Magnitude'] = pd.to_numeric(acc_df['Magnitude'], errors='coerce')

        # Sort heart rate value by date time
        hr_df = hr_df.sort_values(by='datetime')
        acc_df = acc_df.sort_values(by='datetime')

        # Initialize a new DataFrame for the time series
        time_series_df = pd.DataFrame(index=bg_df.index)  # Use the glucose timestamps as the index

        # Create new columns for glucose values at t-0, t-5, t-10, ..., t-30 minutes
        time_window = [60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]  # Minutes

        # Loop through the time window and create the columns for glucose values
        for minutes in time_window:
            shifted_col = bg_df['Glucose Value (mg/dL)'].shift(periods=minutes // 5)  # 5-minute intervals
            time_series_df[f'Glucose (t-{minutes})'] = shifted_col

        time_series_df.dropna(inplace=True)

        # Define the time window for heart rate and accelerometer data
        # time_window = [60, 50, 40, 30, 20, 10]  # Added t-0 for current values

        # For each glucose timestamp, find the closest heart rate and accelerometer readings
        for minutes in time_window[:-1]:
            # Calculate the lookback time delta
            lookback = pd.Timedelta(minutes=minutes)
            
            # For each glucose timestamp
            for idx, row in time_series_df.iterrows():
                # Find the closest heart rate reading within a certain time window
                target_time = idx - lookback
                closest_hr = hr_df[' hr'].asof(target_time)  # Gets the most recent value before target_time
                time_series_df.at[idx, f'Heart Rate (t-{minutes})'] = closest_hr
                
                # Find the closest accelerometer magnitude
                closest_acc = acc_df['Magnitude'].asof(target_time)
                time_series_df.at[idx, f'Magnitude (t-{minutes})'] = closest_acc

        # time_window = [60, 30]
        
        # Initialize food-related columns explicitly as float
        for minutes in time_window[:-1]:
            time_series_df[f'Calories (t-{minutes})'] = 0.0  # Use 0.0 to ensure float type
            time_series_df[f'Total Carbs (t-{minutes})'] = 0.0
            time_series_df[f'Sugar (t-{minutes})'] = 0.0
        
        for minutes in time_window[:-1]:
            # For each glucose timestamp
            for idx, row in time_series_df.iterrows():
                # Define the time window for food intake
                start_time = idx - pd.Timedelta(minutes=minutes)
                end_time = idx
                
                # Filter food entries within the time window
                # food_in_window = food_df.loc[start_time:end_time]
                food_in_window = food_df[(food_df['time_begin'] >= start_time) & (food_df['time_begin'] <= end_time)]

                # Calculate cumulative values
                if not food_in_window.empty:
                    time_series_df.at[idx, f'Calories (t-{minutes})'] = float(food_in_window['calorie'].sum())
                    time_series_df.at[idx, f'Total Carbs (t-{minutes})'] = float(food_in_window['total_carb'].sum())
                    time_series_df.at[idx, f'Sugar (t-{minutes})'] = float(food_in_window['sugar'].sum())
                else:
                    time_series_df.at[idx, f'Calories (t-{minutes})'] = 0.0  # Use 0.0 instead of 0
                    time_series_df.at[idx, f'Total Carbs (t-{minutes})'] = 0.0
                    time_series_df.at[idx, f'Sugar (t-{minutes})'] = 0.0

        patient_demographics = demographic_data[demographic_data['ID'] == i]
        
        if not patient_demographics.empty:
            # Add demographic information to every row
            time_series_df['Gender'] = patient_demographics['Gender'].values[0]
            time_series_df['HbA1c'] = patient_demographics['HbA1c'].values[0]

        # Save with index (which contains the timestamps)
        path = '../data/processed/dataset_by_patient/patient_'+patient+'.csv'
        time_series_df.to_csv(path)

    return
    
def create_dictionary():
    data_path = '../data/processed/dataset_by_patient'

    time_window = [60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]  # Minutes

    dataset_dic = dict()    
    dataset_dic = {
        'patient_id': [],
        'glucose_inputs': [],
        'accelerometer': [],
        'heart_rate': [],
        'calories': [],
        'sugar': [],
        'carbs': [],
        'glucose outputs': [],
        'time_window': time_window[:-1],
        'gender': [],
        'HbA1c': []
        }

    for i in range(1,17):
        if i<10:
            patient = "00"+str(i)
        else:
            patient = "0"+str(i)

        print("Patient"+str(i))
                
        # Load files
        df = pd.read_csv(f"{data_path}/patient_{patient}.csv")

        # Fill dictionary
        for j in range(len(df)):
        
            glucose_columns = [col for col in df.columns if "Glucose" in col and col != "Glucose (t-10)"]
            glucose_values = df.loc[j, glucose_columns].tolist()
            
            accelerometer_columns = [col for col in df.columns if "Magnitude" in col and col != "Glucose (t-10)"]
            accelerometer_values = df.loc[j, accelerometer_columns].tolist()

            heart_rate_columns = [col for col in df.columns if "Heart Rate " in col and col != "Glucose (t-10)"]
            heart_rate_values = df.loc[j, heart_rate_columns].tolist()

            calories_columns = [col for col in df.columns if "Calories" in col and col != "Glucose (t-10)"]
            calories_values = df.loc[j, calories_columns].tolist()

            sugar_columns = [col for col in df.columns if "Sugar" in col and col != "Glucose (t-10)"]
            sugar_values = df.loc[j, sugar_columns].tolist()

            carbs_columns = [col for col in df.columns if "Carbs" in col and col != "Glucose (t-10)"]
            carbs_values = df.loc[j, carbs_columns].tolist()

            outputs_columns = [col for col in df.columns if "Glucose (t-0)" in col and col != "Glucose (t-10)"]
            outputs_values = df.loc[j, outputs_columns].tolist()

            dataset_dic['patient_id'].append(i)
            dataset_dic['glucose_inputs'].append(glucose_values)
            dataset_dic['accelerometer'].append(accelerometer_values)
            dataset_dic['heart_rate'].append(heart_rate_values)
            dataset_dic['calories'].append(calories_values)
            dataset_dic['sugar'].append(sugar_values)
            dataset_dic['carbs'].append(carbs_values)
            dataset_dic['glucose outputs'].append(outputs_values)
            dataset_dic['gender'].append(df.loc[j,'Gender'])
            dataset_dic['HbA1c'].append(df.loc[j,'HbA1c'])


    with open("../data/processed/data.json", "w") as f:
        json.dump(dataset_dic, f, indent=4)  # indent=4 makes it more readable

    return


# create_dataframes()
create_dictionary()