import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

PLOT_FEATURES = False

features = pd.read_csv('features_and_ground_truth.csv', index_col=0)

if PLOT_FEATURES:
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

labels = np.array(features[['FBDRRMean(Ch)', 'FBT60Mean(Ch)']])

#Drop label columns and string columns
features = features.drop(['FBDRRMean(Ch)', 'FBT60Mean(Ch)', 'file', 'filename'], axis=1)

feature_list = list(features.columns)

features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                            random_state=42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

rf = RandomForestRegressor(n_estimators=5000, random_state=42)
rf.fit(train_features, train_labels)

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
