import pandas as pd
from os import path
from glob import glob
import numpy as np

DATA_PATH = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Eval/Speech/'

features = pd.read_csv(path.join(DATA_PATH, 'audio_features.csv'))

# Add empty ground truth columns
features['FBDRRMean(Ch)'] = np.nan
features['FBT60Mean(Ch)'] = np.nan

for i, row in features.iterrows():
    mic_type = row['filename'].split('_')[0]
    # The following solution is not elegant.
    if row['filename'].find('1_M') != -1:
        target_name = row['filename'].split('1_M')[0]
    elif row['filename'].find('2_M') != -1:
        target_name = row['filename'].split('2_M')[0]
    elif row['filename'].find('1_F') != -1:
        target_name = row['filename'].split('1_F')[0]
    else:
        target_name = row['filename'].split('2_F')[0]
    ground_truth_csv_path = glob(DATA_PATH + mic_type + '/*reduced.csv')[0]
    ground_truth_csv = pd.read_csv(ground_truth_csv_path)
    ground_truth_column = ground_truth_csv[ground_truth_csv['Filename'].str.contains(target_name)]
    features.at[i, 'FBDRRMean(Ch)'] = ground_truth_column['FBDRRMean(Ch)']
    features.at[i, 'FBT60Mean(Ch)'] = ground_truth_column['FBT60Mean(Ch)']

features.to_csv(DATA_PATH + 'features_and_ground_truth.csv')
