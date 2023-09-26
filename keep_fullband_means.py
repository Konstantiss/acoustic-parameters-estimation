import pandas as pd
import os
import os.path as path
from pathlib import Path

DATA_PATH = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Dev/Speech'

# Keep FB T60 Mean (Ch) and FB DRR Mean (Ch) values for each Room / Configuration

for dir, subDirs, files in os.walk(DATA_PATH):
    for subDir in subDirs:
        all_files = os.listdir(path.join(dir, subDir))
        csv_name = list(filter(lambda f: f.endswith('.csv'), all_files))
        csv_name = csv_name[0]
        csv_path = path.join(path.join(dir, subDir), csv_name)
        df = pd.read_csv(csv_path)
        #Remove spaces from column names
        df.columns = df.columns.str.replace(' ', '')
        #Remove ":" from column names
        df.columns = df.columns.str.replace(':', '')
        df_reduced = df.drop_duplicates(subset=['Filename'], keep="first")
        output_csv_path = csv_path.replace(".csv", "_reduced.csv")
        df_reduced.to_csv(output_csv_path)
