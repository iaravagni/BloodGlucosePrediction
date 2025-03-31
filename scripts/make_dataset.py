import pandas as pd
import numpy as np
import json


def clean_blood_glucose_df(bg_df):
    # Filter the rows where Event Type is 'EGV'
    bg_df = bg_df[bg_df['Event Type'] == 'EGV']
    return bg_df

def get_accelerometer_values(acc_df, time_series_df, window_size='1h'):
    # Calculate magnitude for accelerometer data
    acc_df['Magnitude'] = np.sqrt(acc_df[' acc_x']**2 + acc_df[' acc_y']**2 + acc_df[' acc_z']**2).round(2)
    acc_df['Magnitude'] = pd.to_numeric(acc_df['Magnitude'], errors='coerce')

    weighted_avgs = []
    window_timedelta = pd.Timedelta(window_size)
    
    for ts in time_series_df['Timestamp']:
        # Select only accelerometer data within the time window
        relevant_acc = acc_df[(acc_df['Timestamp'] >= ts - window_timedelta) & (acc_df['Timestamp'] <= ts)]
        
        if not relevant_acc.empty:
            # Compute weighted average: more recent values have higher weight
            time_diffs = (ts - relevant_acc['Timestamp']).dt.total_seconds()
            weights = 1 / (time_diffs + 1)  # Avoid division by zero
            weighted_avg = ((relevant_acc['Magnitude'] * weights).sum() / weights.sum()).round(2)
        else:
            weighted_avg = 0
        
        weighted_avgs.append(weighted_avg)

    
    time_series_df['Accelerometer'] = weighted_avgs
    
    return time_series_df

def get_food_values(food_df, time_series_df, window_size='1h'):
    # Initialize arrays for food metrics
    calories = []
    carbs = []
    sugar = []
    
    window_timedelta = pd.Timedelta(window_size)
    
    for ts in time_series_df['Timestamp']:
        # Select only food data within the time window
        food_in_window = food_df[(food_df['Timestamp'] >= ts - window_timedelta) & 
                                 (food_df['Timestamp'] <= ts)]
        
        # Calculate cumulative values
        if not food_in_window.empty:
            calories.append(food_in_window['calorie'].sum())
            carbs.append(food_in_window['total_carb'].sum())
            sugar.append(food_in_window['sugar'].sum())
        else:
            calories.append(0.0)
            carbs.append(0.0)
            sugar.append(0.0)
    
    # Add to time series dataframe
    time_series_df['Calories'] = calories
    time_series_df['Carbs'] = carbs
    time_series_df['Sugar'] = sugar
    
    return time_series_df

def calculate_age(born, as_of_date=pd.Timestamp('2019-01-01')):
    
    born = pd.Timestamp(born)

    # Calculate age
    age = as_of_date.year - born.year
    
    return age

def split_train_test_patients(df, seed=42):
    np.random.seed(seed)
    training_patients = np.random.choice(np.arange(1, 16), size=13, replace=False)

    test_patients = np.setdiff1d(np.arange(1, 16), training_patients)

    validation_patients = np.random.choice(training_patients, size=2, replace=False)

    training_patients = np.setdiff1d(training_patients, validation_patients)

    df_train = df[df['patient_id'].isin(training_patients)]
    df_val = df[df['patient_id'].isin(validation_patients)]
    df_test = df[df['patient_id'].isin(test_patients)]

    return df_train, df_val, df_test


def create_dataframes():
    data_path = f"./data/raw/big_ideas_dataset"

    for i in range(1,17):
        patient = f"{i:03d}"

        print("Patient"+str(i))
                
        # Load files
        bg_df = pd.read_csv(f"{data_path}/{patient}/Dexcom_{patient}.csv")
        acc_df = pd.read_csv(f"{data_path}/{patient}/ACC_{patient}.csv")
        food_df = pd.read_csv(f"{data_path}/{patient}/Food_Log_{patient}.csv")
        demographic_data = pd.read_csv(f"{data_path}/Demographics.csv")

        # Clean and convert 'Timestamp' columns to datetime format
        bg_df['Timestamp'] = pd.to_datetime(bg_df['Timestamp (YYYY-MM-DDThh:mm:ss)'], errors='coerce')
        acc_df['Timestamp'] = pd.to_datetime(acc_df['datetime'], errors='coerce')
        food_df['Timestamp'] = pd.to_datetime(food_df['time_begin'], errors='coerce')
        
        # Sort values by date time
        bg_df = bg_df.sort_values(by='Timestamp')
        acc_df = acc_df.sort_values(by='Timestamp')


        # Reset index and then find the row where 'Event Type' is 'DateOfBirth'
        reset_df = bg_df.reset_index(drop=True)
        patient_dob = reset_df[reset_df['Event Type'] == 'DateOfBirth']['Patient Info'].values[0]

        patient_age = calculate_age(patient_dob)

        bg_df = clean_blood_glucose_df(bg_df)
        
        # Initialize a new DataFrame for the time series
        time_series_df = pd.DataFrame(index=bg_df.index)  # Use the glucose timestamps as the index

        time_series_df[['Timestamp','Glucose']] = bg_df[['Timestamp','Glucose Value (mg/dL)']]

        # time_series_df = get_acc_hr_values(acc_df, hr_df, time_series_df)
        time_series_df = get_accelerometer_values(acc_df, time_series_df)
        time_series_df = get_food_values(food_df,time_series_df)

        patient_demographics = demographic_data[demographic_data['ID'] == i]

        gender = patient_demographics['Gender'].values[0]  # Assuming you want the first value
        time_series_df['Gender'] = np.where(gender == 'FEMALE', 1, 0)
        time_series_df['HbA1c'] = patient_demographics['HbA1c'].values[0]
        time_series_df['Age'] = patient_age

        path = './data/processed/dataset_by_patient/patient_'+patient+'.csv'
        time_series_df.to_csv(path)

    return


def combine_dataframes():
    data_path = f"./data/processed/dataset_by_patient/patient_"
    combined_df = pd.DataFrame()

    for i in range(1, 17):
        patient = f"{i:03d}"

        print(f"Patient {i}")

        current_df = pd.read_csv(f"{data_path}{patient}.csv")

        current_df["patient_id"] = i

        combined_df = pd.concat([combined_df, current_df], ignore_index=True)

    combined_df = combined_df.iloc[:, 1:]

    df_train, df_val, df_test = split_train_test_patients(combined_df)

    path = './data/processed/'
    combined_df.to_csv(f'{path}combined_dataset.csv')
    df_train.to_csv(f'{path}train_dataset.csv')
    df_val.to_csv(f'{path}validation_dataset.csv')
    df_test.to_csv(f'{path}test_dataset.csv')

    return

def main():
    print("Running make_dataset script...")
    create_dataframes()
    combine_dataframes()

    return

if __name__ == '__main__':
    main()
