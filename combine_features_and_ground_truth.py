import pandas as pd
from os import path
from glob import glob
import numpy as np

DATA_PATH = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Eval/Speech/'

features = pd.read_csv('audio_features_eval.csv')

ROOMS_AND_MIC_CONFIGS = ['Office_1_1', 'Office_1_2', 'Office_2_1', 'Office_2_2', 'Meeting_Room_1_1', 'Meeting_Room_1_2',
                         'Meeting_Room_1_2', 'Meeting_Room_2_1', 'Meeting_Room_2_2', 'Lecture_Room_1_1',
                         'Lecture_Room_1_2',
                         'Lecture_Room_2_1', 'Lecture_Room_2_2', 'Building_Lobby_1', 'Building_Lobby_2']

# Add empty ground truth columns
features['FBDRRMean(Ch)'] = np.nan
features['FBT60Mean(Ch)'] = np.nan

for i, row in features.iterrows():
    mic_type = row['filename'].split('_')[0]
    room_and_mic_config = [substring for substring in ROOMS_AND_MIC_CONFIGS if substring in row['filename']][0]
    target_name = mic_type + '_' + room_and_mic_config
    ground_truth_csv_path = glob(DATA_PATH + mic_type + '/*reduced.csv')[0]
    ground_truth_csv = pd.read_csv(ground_truth_csv_path)
    ground_truth_column = ground_truth_csv[ground_truth_csv['Filename'].str.contains(target_name)]
    features.at[i, 'FBDRRMean(Ch)'] = ground_truth_column['FBDRRMean(Ch)']
    features.at[i, 'FBT60Mean(Ch)'] = ground_truth_column['FBT60Mean(Ch)']

features.to_csv(DATA_PATH + 'features_and_ground_truth_eval.csv')
