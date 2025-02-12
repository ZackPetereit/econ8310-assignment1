import pandas as pd
from pygam import LinearGAM, s


# Load the training data
train_data = pd.read_csv('assignment_data_train.csv', parse_dates=['Timestamp'], index_col='Timestamp')

# Resample data to hourly frequency, summing the number of trips
train_data = train_data.resample('H').sum()

# Step 2: Define the forecasting model
X = train_data.index.hour.values.reshape(-1, 1)
y = train_data['trips']

# Initialize and fit the model
model = LinearGAM(s(0)).fit(X, y)
modelFit = model

# Step 3: Forecasting
test_data = pd.read_csv('assignment_data_test.csv', parse_dates=['Timestamp'], index_col='Timestamp')
test_data = test_data.resample('H').sum()

X_test = test_data.index.hour.values.reshape(-1, 1)
pred = model.predict(X_test)

# Save predictions to a CSV file (optional)
prediction_df = pd.DataFrame(data=pred, index=test_data.index, columns=['predicted_number_of_trips'])
prediction_df.to_csv('predictions.csv')
