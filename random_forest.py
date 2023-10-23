import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, accuracy_score, mean_squared_error, r2_score, mean_absolute_error

PLOT_FEATURES = False

features = pd.read_csv('features_and_ground_truth_eval.csv', index_col=0)

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

# Drop label columns and string columns
features = features.drop(['FBDRRMean(Ch)', 'FBT60Mean(Ch)', 'file', 'filename'], axis=1)

feature_list = list(features.columns)

features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                            random_state=42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

rf = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1, verbose=2)
rf.fit(train_features, train_labels)
print("Random forest parameters: ", rf.get_params())


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    r_squared = r2_score(test_labels, predictions)
    mae = mean_absolute_error(test_labels, predictions)
    mse = mean_squared_error(test_labels, predictions)
    mape = mean_absolute_percentage_error(test_labels, predictions)
    accuracy = 100 - mape

    print('------Model Performance------')
    print('R squared:', r_squared)
    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', np.sqrt(mse))
    # print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    # print('Accuracy:', round(100 * accuracy, 2))

evaluate(rf, test_features, test_labels)

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

PLOT_VARIABLE_IMPORTANCES = False

if PLOT_VARIABLE_IMPORTANCES:
    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.bar(x_values, importances, orientation='vertical')
    # Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation='horizontal', size=10)
    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    plt.show()

GRID_SEARCH = False

if GRID_SEARCH:
    param_grid = {
        'bootstrap': [True],
        'max_features': [1, 7, 9, 10, 11],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [2, 3, 4],
        'n_estimators': [1000],
        'random_state': [40, 45, 50, 55]
    }
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=-1)
    grid_search.fit(train_features, train_labels)

    print("Best parameters: ", grid_search.best_params_)

    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, test_features, test_labels)
