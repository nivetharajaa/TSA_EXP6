### DEVELOPED BY:Nivetha A
### REGISTER NO: 212222230101
### Date:

# Ex.No: 6               HOLT WINTERS METHOD


### AIM:
To forecast sales using the Holt-Winters method and calculate the Test and Final Predictions.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'raw_sales.csv'  # Path to the uploaded dataset
data = pd.read_csv(file_path)

# Convert 'datesold' to datetime format
data['datesold'] = pd.to_datetime(data['datesold'])

# Group data by date and resample it to month-end frequency ('ME')
monthly_data = data.resample('ME', on='datesold').sum()

# Plot the time series data
plt.figure(figsize=(10, 5))
plt.plot(monthly_data['price'], label='Monthly Sales Data')
plt.title('Monthly Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales Price')
plt.legend()
plt.show()

# Split data into training and testing sets (80% for training, 20% for testing)
train_size = int(len(monthly_data) * 0.8)
train, test = monthly_data['price'][:train_size], monthly_data['price'][train_size:]

# Fit the Holt-Winters model on training data
model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit()

# Make predictions on the test set
predictions = fit.forecast(len(test))

# Calculate RMSE for the test set predictions
rmse = np.sqrt(mean_squared_error(test, predictions))
print(f'Test RMSE: {rmse}')

# Fit Holt-Winters model on the entire dataset for future forecasting
final_model = ExponentialSmoothing(monthly_data['price'], trend="add", seasonal="add", seasonal_periods=12)
final_fit = final_model.fit()

# Make future predictions (for 12 months)
future_steps = 12
final_forecast = final_fit.forecast(steps=future_steps)

# Plotting Test Predictions and Final Predictions
plt.figure(figsize=(12, 6))

# Plot Test Predictions
plt.subplot(1, 2, 1)
plt.plot(monthly_data.index[:train_size], train, label='Training Data', color='blue')
plt.plot(monthly_data.index[train_size:], test, label='Test Data', color='green')
plt.plot(monthly_data.index[train_size:], predictions, label='Predictions', color='orange')
plt.title('Test Predictions')
plt.xlabel('Date')
plt.ylabel('Sales Price')
plt.legend()

# Plot Final Predictions
plt.subplot(1, 2, 2)
plt.plot(monthly_data.index, monthly_data['price'], label='Original Sales Data', color='blue')
# Plot future forecast (use 'ME' frequency)
plt.plot(pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), periods=future_steps, freq='ME'), 
         final_forecast, label='Final Forecast', color='orange')
plt.title('Final Predictions')
plt.xlabel('Date')
plt.ylabel('Sales Price')
plt.legend()

plt.tight_layout()
plt.show()
```

### OUTPUT:


### TEST AND FINAL_PREDICTION

![376123796-d308b74e-2274-4254-b55c-3f2aa0342426](https://github.com/user-attachments/assets/5c509e16-416c-404f-9acf-cb0010fea44b)



### RESULT:
Thus, the program to forecast sales using the Holt-Winters method and calculate the Test and Final Prediction is executed successfully.
