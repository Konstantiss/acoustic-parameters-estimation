import pandas as pd
import matplotlib.pyplot as plt

features = pd.read_csv('features_and_ground_truth.csv', index_col=0)

features['spectral_centroid'].plot(title='Spectral Centroid')
plt.show()

features['spectral_bandwidth'].plot(title='Spectral Bandwidth')
plt.show()

features['spectral_rolloff'].plot(title='Spectral Rolloff')
plt.show()

features['rmse'].plot(title='RMSE')
plt.show()

features['zero_crossing_rate'].plot(title='Zero Crossing Rate')
plt.show()


