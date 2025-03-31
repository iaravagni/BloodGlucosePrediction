from collections import defaultdict
import pandas as pd
import json

def get_batched_data_fn(
    df,
    batch_size: int = 128, 
    context_len: int = int(60/5), # 1 hour, sampled every 5 mins 
    horizon_len: int = int(30/5), # 30 secs, sampled every 5 mins
):
  examples = defaultdict(list)

  num_examples = 0
  for patient in range(1,17):
    sub_df = df[df["patient_id"] == patient] 
    for start in range(0, len(sub_df) - (context_len + horizon_len), horizon_len):
      num_examples += 1
      examples["patient_id"].append(patient)
      examples["gender"].append(sub_df.iloc[0]["Gender"])
      examples["HbA1c"].append(sub_df.iloc[0]['HbA1c'])
      examples["inputs"].append(sub_df["Glucose"][start:(context_end := start + context_len)].tolist())
      examples["accelerometer"].append(sub_df["Accelerometer"][start:context_end + horizon_len].tolist())
      examples["calories"].append(sub_df["Calories"][start:context_end + horizon_len].tolist())
      examples["sugar"].append(sub_df["Sugar"][start:context_end + horizon_len].tolist())
      examples["carbs"].append(sub_df["Carbs"][start:context_end + horizon_len].tolist())
      examples["timestamp"].append(sub_df["Timestamp"][start:context_end + horizon_len].tolist())
      examples["outputs"].append(sub_df["Glucose"][context_end:(context_end + horizon_len)].tolist())
  
  def data_fn():
    for i in range(1 + (num_examples - 1) // batch_size):
      yield {k: v[(i * batch_size) : ((i + 1) * batch_size)] for k, v in examples.items()}
  
  return data_fn

dataframe = pd.read_csv('../data/processed/combined_dataset.csv')
input_data = get_batched_data_fn(dataframe, batch_size = 128)

with open("../data/processed/combined_dataset.json", "w") as f:
        json.dump(input_data, f, indent=4) 