import librosa
import librosa.display
import os.path as path
import IPython.display as ipd
from playsound import playsound
import matplotlib.pyplot as plt

WAV_FILES_PATH = '/home/konstantis/Nextcloud/ΤΗΜΜΥ/Thesis/Data/ACE/script-output/Dev/Speech'

audio_data = path.join(WAV_FILES_PATH, 'Single/Single_Office_1_1_M8_s3_Fan_0dB.wav')
x, sr = librosa.load(audio_data)

#playsound(audio_data)

plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
plt.show()

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()
