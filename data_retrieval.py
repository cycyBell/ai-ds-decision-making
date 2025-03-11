import numpy as np
import pandas as pd
import gzip

def get_models_data(file_path) : 

    with gzip.open(file_path, "rt") as f:
        dataset_2d = pd.read_csv(f)

    rounded_dataset_2d = dataset_2d.to_numpy().round(2)
    dataset = rounded_dataset_2d.reshape(-1, 10, 12)

    ds_ahp_dataset = rounded_dataset_2d.reshape(-1,10)*100
    pca_dataset = rounded_dataset_2d.reshape(-1,12)
    vae_dataset = np.expand_dims(dataset, axis = -1)
    
    return {
       "ds_ahp_data" : ds_ahp_dataset,
       "pca_data" : pca_dataset,
       "vae_data" : vae_dataset
    }

def get_alternatives_matrix(ds_ahp_dataset, criteria_list):
  criteria_number = len(criteria_list)
  data = dict()
  start = 0
  stop = 45000
  for criterion in criteria_list:
    pw_matrix = ds_ahp_dataset[start:stop, :]
    start += 45000
    stop += 45000
    data[criterion] = pw_matrix
  return data