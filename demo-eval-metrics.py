import numpy as np

data_gt = np.load('training_ground_truth_data.npz', allow_pickle=True)
training_data = data_gt['training_data']
ground_truth = data_gt['ground_truth']
data_forecast = np.load('forecasts.npz', allow_pickle=True)
forecast = data_forecast['forecasts']
print('forecast:')
print(forecast[0][0])

print('\ntrain: ')
print(training_data[0])
print('\ntrue:')
print(ground_truth[0])
