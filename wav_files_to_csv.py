import os
import librosa
from librosa import feature
import numpy as np
import csv

DATA_PATH = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Dev/Speech'


def walk(dir):
    files = []
    for p, d, f in os.walk(dir):
        for file in f:
            if file.endswith('.wav'):
                files.append(p + '/' + file)
    return files


audio_files = walk(DATA_PATH)

fn_list_i = [
    feature.chroma_stft,
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.spectral_rolloff
]

fn_list_ii = [
    feature.rms,
    feature.zero_crossing_rate
]


def get_feature_vector(y, sr):
    feat_vect_i = [np.mean(funct(y=y, sr=sr)) for funct in fn_list_i]
    feat_vect_ii = [np.mean(funct(y=y)) for funct in fn_list_ii]
    feature_vector = feat_vect_i + feat_vect_ii
    return feature_vector


audio_features = []
for file in audio_files:
    y, sr = librosa.load(file, sr=None)
    feature_vector = get_feature_vector(y, sr)
    audio_features.append(feature_vector)

output_csv = 'audio_features.csv'

header = [
    'file',
    'chroma_stft',
    'spectral_centroid',
    'spectral_bandwidth',
    'spectral_rolloff',
    'rmse',
    'zero_crossing_rate'
]

# Append the file names to the audio features list that will be written to the csv
index = 0
for sublist in audio_features:
    sublist.insert(0, audio_files[index])
    index += 1

with open(output_csv, '+w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerow(header)
    csv_writer.writerows(audio_features)
