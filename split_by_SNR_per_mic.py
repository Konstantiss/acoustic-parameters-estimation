import pandas as pd

DATA_PATH_EVAL = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Eval/Speech/'
annotations_file_path_eval = DATA_PATH_EVAL + 'features_and_ground_truth_eval.csv'

DATA_PATH_TRAIN = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Train/Speech/'
annotations_file_path_train = DATA_PATH_TRAIN + 'features_and_ground_truth_train.csv'

filenames_eval = ['single_eval.csv', 'chromebook_eval.csv', 'mobile_eval.csv', 'crucif_eval.csv', 'lin8ch_eval.csv',
                  'em32_eval.csv']
filenames_train = ['single_train.csv', 'chromebook_train.csv', 'mobile_train.csv', 'crucif_train.csv',
                   'lin8ch_train.csv',
                   'em32_train.csv']

data_eval = pd.read_csv(annotations_file_path_train)
data_train = pd.read_csv(annotations_file_path_train)

def split(filenames, train=False):
    dataframes = [[None, None, None], [None, None, None], [None, None, None], [None, None, None], [None, None, None],
                  [None, None, None]]

    df = pd.DataFrame()
    mic_index = 0
    for filename in filenames:

        if train:
            df = pd.read_csv(DATA_PATH_TRAIN + filename)
        else:
            df = pd.read_csv(DATA_PATH_EVAL + filename)

        low_snr = pd.DataFrame(columns=df.columns.values)
        mid_snr = pd.DataFrame(columns=df.columns.values)
        high_snr = pd.DataFrame(columns=df.columns.values)

        if train:
            for i, row in df.iterrows():
                if '_-1dB' in row['filename']:
                    low_snr.loc[len(low_snr)] = row
                if '_12dB' in row['filename']:
                    mid_snr.loc[len(mid_snr)] = row
                if '_18dB' in row['filename']:
                    high_snr.loc[len(high_snr)] = row
        else:
            for i, row in df.iterrows():
                if '_0dB' in row['filename']:
                    low_snr.loc[len(low_snr)] = row
                if '_10dB' in row['filename']:
                    mid_snr.loc[len(mid_snr)] = row
                if '_20dB' in row['filename']:
                    high_snr.loc[len(high_snr)] = row

        dataframes[mic_index][0] = low_snr
        dataframes[mic_index][1] = mid_snr
        dataframes[mic_index][2] = high_snr

        mic_index += 1

    return dataframes


dataframes_eval = split(filenames_eval, train=False)

single_low_snr = dataframes_eval[0][0]
single_low_snr.to_csv(DATA_PATH_EVAL + 'single_0dB_eval.csv')
single_mid_snr = dataframes_eval[0][1]
single_mid_snr.to_csv(DATA_PATH_EVAL + 'single_10dB_eval.csv')
single_high_snr = dataframes_eval[0][2]
single_high_snr.to_csv(DATA_PATH_EVAL + 'single_20dB_eval.csv')
chromebook_low_snr = dataframes_eval[1][0]
chromebook_low_snr.to_csv(DATA_PATH_EVAL + 'chromebook_0dB_eval.csv')
chromebook_mid_snr = dataframes_eval[1][1]
chromebook_mid_snr.to_csv(DATA_PATH_EVAL + 'chromebook_10dB_eval.csv')
chromebook_high_snr = dataframes_eval[1][2]
chromebook_high_snr.to_csv(DATA_PATH_EVAL + 'chromebook_20dB_eval.csv')
mobile_low_snr = dataframes_eval[2][0]
mobile_low_snr.to_csv(DATA_PATH_EVAL + 'mobile_0dB_eval.csv')
mobile_mid_snr = dataframes_eval[2][1]
mobile_mid_snr.to_csv(DATA_PATH_EVAL + 'mobile_10dB_eval.csv')
mobile_high_snr = dataframes_eval[2][2]
mobile_high_snr.to_csv(DATA_PATH_EVAL + 'mobile_20dB_eval.csv')
crucif_low_snr = dataframes_eval[3][0]
crucif_low_snr.to_csv(DATA_PATH_EVAL + 'crucif_0dB_eval.csv')
crucif_mid_snr = dataframes_eval[3][1]
crucif_mid_snr.to_csv(DATA_PATH_EVAL + 'crucif_10dB_eval.csv')
crucif_high_snr = dataframes_eval[3][2]
crucif_high_snr.to_csv(DATA_PATH_EVAL + 'crucif_20dB_eval.csv')
lin8ch_low_snr = dataframes_eval[4][0]
lin8ch_low_snr.to_csv(DATA_PATH_EVAL + 'lin8ch_0dB_eval.csv')
lin8ch_mid_snr = dataframes_eval[4][1]
lin8ch_mid_snr.to_csv(DATA_PATH_EVAL + 'lin8ch_10dB_eval.csv')
lin8ch_high_snr = dataframes_eval[4][2]
lin8ch_high_snr.to_csv(DATA_PATH_EVAL + 'lin8ch_20dB_eval.csv')
em32_low_snr = dataframes_eval[5][0]
em32_low_snr.to_csv(DATA_PATH_EVAL + 'em32_0dB_eval.csv')
em32_mid_snr = dataframes_eval[5][1]
em32_mid_snr.to_csv(DATA_PATH_EVAL + 'em32_10dB_eval.csv')
em32_high_snr = dataframes_eval[5][2]
em32_high_snr.to_csv(DATA_PATH_EVAL + 'em32_20dB_eval.csv')

_dataframes_train = split(filenames_train, train=True)

single_low_snr = _dataframes_train[0][0]
single_low_snr.to_csv(DATA_PATH_TRAIN + 'single_-1dB_train.csv')
single_mid_snr = _dataframes_train[0][1]
single_mid_snr.to_csv(DATA_PATH_TRAIN + 'single_12dB_train.csv')
single_high_snr = _dataframes_train[0][2]
single_high_snr.to_csv(DATA_PATH_TRAIN + 'single_18dB_train.csv')
chromebook_low_snr = _dataframes_train[1][0]
chromebook_low_snr.to_csv(DATA_PATH_TRAIN + 'chromebook_-1dB_train.csv')
chromebook_mid_snr = _dataframes_train[1][1]
chromebook_mid_snr.to_csv(DATA_PATH_TRAIN + 'chromebook_12dB_train.csv')
chromebook_high_snr = _dataframes_train[1][2]
chromebook_high_snr.to_csv(DATA_PATH_TRAIN + 'chromebook_18dB_train.csv')
mobile_low_snr = _dataframes_train[2][0]
mobile_low_snr.to_csv(DATA_PATH_TRAIN + 'mobile_-1dB_train.csv')
mobile_mid_snr = _dataframes_train[2][1]
mobile_mid_snr.to_csv(DATA_PATH_TRAIN + 'mobile_12dB_train.csv')
mobile_high_snr = _dataframes_train[2][2]
mobile_high_snr.to_csv(DATA_PATH_TRAIN + 'mobile_18dB_train.csv')
crucif_low_snr = _dataframes_train[3][0]
crucif_low_snr.to_csv(DATA_PATH_TRAIN + 'crucif_-1dB_train.csv')
crucif_mid_snr = _dataframes_train[3][1]
crucif_mid_snr.to_csv(DATA_PATH_TRAIN + 'crucif_12dB_train.csv')
crucif_high_snr = _dataframes_train[3][2]
crucif_high_snr.to_csv(DATA_PATH_TRAIN + 'crucif_18dB_train.csv')
lin8ch_low_snr = _dataframes_train[4][0]
lin8ch_low_snr.to_csv(DATA_PATH_TRAIN + 'lin8ch_-1dB_train.csv')
lin8ch_mid_snr = _dataframes_train[4][1]
lin8ch_mid_snr.to_csv(DATA_PATH_TRAIN + 'lin8ch_12dB_train.csv')
lin8ch_high_snr = _dataframes_train[4][2]
lin8ch_high_snr.to_csv(DATA_PATH_TRAIN + 'lin8ch_18dB_train.csv')
em32_low_snr = _dataframes_train[5][0]
em32_low_snr.to_csv(DATA_PATH_TRAIN + 'em32_-1dB_train.csv')
em32_mid_snr = _dataframes_train[5][1]
em32_mid_snr.to_csv(DATA_PATH_TRAIN + 'em32_12dB_train.csv')
em32_high_snr = _dataframes_train[5][2]
em32_high_snr.to_csv(DATA_PATH_TRAIN + 'em32_18dB_train.csv')
