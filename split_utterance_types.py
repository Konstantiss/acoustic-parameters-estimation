import pandas as pd

DATA_PATH_EVAL = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Eval/Speech/'
annotations_file_path_eval = DATA_PATH_EVAL + 'features_and_ground_truth_eval.csv'

DATA_PATH_TRAIN = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Train/Speech/'
annotations_file_path_train = DATA_PATH_TRAIN + 'features_and_ground_truth_train.csv'

data_eval = pd.read_csv(annotations_file_path_eval)
data_train = pd.read_csv(annotations_file_path_train)

def split(df):
    s1_utterances = pd.DataFrame(columns=df.columns.values)
    s2_utterances = pd.DataFrame(columns=df.columns.values)
    s3_utterances = pd.DataFrame(columns=df.columns.values)
    s4_utterances = pd.DataFrame(columns=df.columns.values)
    s5_utterances = pd.DataFrame(columns=df.columns.values)

    for i, row in df.iterrows():
        if '_s1_' in row['filename']:
            s1_utterances.loc[len(s1_utterances)] = row
        if '_s2_' in row['filename']:
            s2_utterances.loc[len(s2_utterances)] = row
        if '_s3_' in row['filename']:
            s3_utterances.loc[len(s3_utterances)] = row
        if '_s4_' in row['filename']:
            s4_utterances.loc[len(s4_utterances)] = row
        if '_s5_' in row['filename']:
            s5_utterances.loc[len(s5_utterances)] = row
    return s1_utterances, s2_utterances, s3_utterances, s4_utterances, s5_utterances

s1_utterances_eval, s2_utterances_eval, s3_utterances_eval, s4_utterances_eval, s5_utterances_eval = split(data_eval)

# In the eval dataset, only s3, s4 utterances are used
s3_utterances_eval.to_csv(DATA_PATH_EVAL + 's3_utterances_eval.csv')
s4_utterances_eval.to_csv(DATA_PATH_EVAL + 's4_utterances_eval.csv')

s1_utterances_train, s2_utterances_train, s3_utterances_train, s4_utterances_train, s5_utterances_train = split(data_train)

s1_utterances_train.to_csv(DATA_PATH_TRAIN + 's1_utterances_train.csv')
s2_utterances_train.to_csv(DATA_PATH_TRAIN + 's2_utterances_train.csv')
s3_utterances_train.to_csv(DATA_PATH_TRAIN + 's3_utterances_train.csv')
s4_utterances_train.to_csv(DATA_PATH_TRAIN + 's4_utterances_train.csv')
s5_utterances_train.to_csv(DATA_PATH_TRAIN + 's5_utterances_train.csv')
