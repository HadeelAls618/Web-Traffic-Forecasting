
import numpy as np
import torch
import matplotlib.pyplot as plt

# Forecasting function
def forecast_next_week(model, last_week_data, days=7):
    model.eval()
    predictions = []
    
    # Ensure the last week data has the correct shape (batch_size, sequence_length, input_size)
    input_seq = torch.tensor(last_week_data).float().unsqueeze(0).to(device)  # Add batch dimension (1, 7, 6)
    
    with torch.no_grad():
        for _ in range(days):
            # Predict for the next day (only one output for unique_visits)
            next_day_pred = model(input_seq).squeeze().item()  # Get the prediction for the next day
            predictions.append(next_day_pred)
            
            # Update the input sequence by removing the oldest day and appending the new prediction
            next_input = np.append(input_seq.cpu().numpy()[0, 1:], [[next_day_pred] * input_seq.shape[2]], axis=0)
            input_seq = torch.tensor(next_input).float().unsqueeze(0).to(device)  # Update with new sequence

    return predictions

# Inverse transform function (for just 'unique_visits' feature)
def inverse_min_max(scaled_values, scaler):
    # Create an array of the same shape as the original data with scaled values for all features
    # We will create a dummy array of 1s for the other features and only modify 'unique_visits' column
    dummy_input = np.ones((scaled_values.shape[0], scaler.scale_.shape[0]))  # Create a dummy array (7, 6)
    dummy_input[:, 3] = scaled_values  # Assuming 'unique_visits' is at index 3 (4th column)

    # Now inverse transform the array that has 'unique_visits' values to the original scale
    inverse_values = scaler.inverse_transform(dummy_input)
    
    # Extract the 'unique_visits' column from the inverse-transformed array
    return inverse_values[:, 3]

# Assuming best_model is your trained LSTM or any model
best_model =  RNNModel(input_size=6, hidden_size=128, output_size=1).to(device)  # Set input_size=6 for all features
best_model.load_state_dict(torch.load("best_rnn_model.pth"))

# Use the last week of data from `X_train` or `X_test`
last_week_data = X_test[-1]  # Last sequence in the test set (should have shape (7, 6))

# Forecast the next 7 days
next_week_forecast = forecast_next_week(best_model, last_week_data)

# Inverse transform the forecasted unique_visits values (forecasted values are only for unique_visits)
forecasted_unique_visits = np.array(next_week_forecast)  # Convert list to numpy array

# Apply inverse transformation
inverse_forecasted = inverse_min_max(forecasted_unique_visits, scaler)

print("Inverse Forecasted Unique Visits for the next week:", inverse_forecasted)

# Now, we will plot the actual data (past week) and the forecasted data (next week)
# Ensure 'data2' has a column 'unique_visits' that you want to compare

# Get the actual data for the past 7 days
past_week_actual = data2['unique_visits'].tail(7).values  # Last 7 days of unique visits in data2

# Get the last date in 'data2'
last_date = data2.index[-1]

# Generate the next 7 days after the last date
forecast_dates = pd.date_range(start=last_date, periods=7, freq='D')  # Exclude the last date, start from the next day

# Assuming 'inverse_forecasted' holds your forecasted values after inverse transformation
forecasted_values = inverse_forecasted  # Adjust this variable as needed
# Plot the actual data (past week) and the forecasted data (next week)
# Plot the actual data (past week) and the forecasted data (next week) with different colors but connected
plt.figure(figsize=(10, 6))

# Plot the actual data (past week) in blue
plt.plot(range(len(past_week_actual)), past_week_actual, label='Actual Unique Visits (Past Week)', marker='o', color='b')

# Plot the forecasted data (next week) in red, continuing from where the actual data ends
plt.plot(range(len(past_week_actual), len(past_week_actual) + len(forecasted_values)), forecasted_values, label='Forecasted Unique Visits (Next Week)', marker='o', color='r')

# Update x-ticks with the last date and the next 7 forecast days
plt.xticks(range(len(past_week_actual) + len(forecasted_values)), 
           list(data2['unique_visits'].tail(7).index) + forecast_dates.strftime('%Y-%m-%d').tolist(), rotation=45)

plt.title("Actual vs. Forecasted Unique Visits for the Next Week")
plt.xlabel("Date")
plt.ylabel("Unique Visits")
plt.legend()

plt.tight_layout()
plt.show()

