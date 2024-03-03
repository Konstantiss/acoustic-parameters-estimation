import pandas as pd

DATA_PATH_EVAL = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Eval/Speech/'
annotations_file_path_eval = DATA_PATH_EVAL + 'features_and_ground_truth_eval.csv'

DATA_PATH_TRAIN = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Train/Speech/'
annotations_file_path_train = DATA_PATH_TRAIN + 'features_and_ground_truth_train.csv'

data_eval = pd.read_csv(annotations_file_path_eval)
data_train = pd.read_csv(annotations_file_path_train)

def split(df):
    single = pd.DataFrame(columns=df.columns.values)
    chromebook = pd.DataFrame(columns=df.columns.values)
    mobile = pd.DataFrame(columns=df.columns.values)
    crucif = pd.DataFrame(columns=df.columns.values)
    lin8ch = pd.DataFrame(columns=df.columns.values)
    em32 = pd.DataFrame(columns=df.columns.values)

    for i, row in df.iterrows():
        if 'Single_' in row['filename']:
            single.loc[len(single)] = row
        if 'Chromebook_' in row['filename']:
            chromebook.loc[len(chromebook)] = row
        if 'Mobile_' in row['filename']:
            mobile.loc[len(mobile)] = row
        if 'Crucif_' in row['filename']:
            crucif.loc[len(crucif)] = row
        if 'Lin8Ch_' in row['filename']:
            lin8ch.loc[len(lin8ch)] = row
        if 'EM32_' in row['filename']:
            em32.loc[len(em32)] = row
    return single, chromebook, mobile, crucif, lin8ch, em32

single_eval, chromebook_eval, mobile_eval, crucif_eval, lin8ch_eval, em32_eval = split(data_eval)

single_eval.to_csv(DATA_PATH_EVAL + 'single_eval.csv')
chromebook_eval.to_csv(DATA_PATH_EVAL + 'chromebook_eval.csv')
mobile_eval.to_csv(DATA_PATH_EVAL + 'mobile_eval.csv')
crucif_eval.to_csv(DATA_PATH_EVAL + 'crucif_eval.csv')
lin8ch_eval.to_csv(DATA_PATH_EVAL + 'lin8ch_eval.csv')
em32_eval.to_csv(DATA_PATH_EVAL + 'em32_eval.csv')

single_train, chromebook_train, mobile_train, crucif_train, lin8ch_train, em32_train = split(data_train)

single_train.to_csv(DATA_PATH_TRAIN + 'single_train.csv')
chromebook_train.to_csv(DATA_PATH_TRAIN + 'chromebook_train.csv')
mobile_train.to_csv(DATA_PATH_TRAIN + 'mobile_train.csv')
crucif_train.to_csv(DATA_PATH_TRAIN + 'crucif_train.csv')
lin8ch_train.to_csv(DATA_PATH_TRAIN + 'lin8ch_train.csv')
em32_train.to_csv(DATA_PATH_TRAIN + 'em32_train.csv')
