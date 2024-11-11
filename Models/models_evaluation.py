
import matplotlib.pyplot as plt
import numpy as np

# Evaluation function
def evaluate_model(model, X_test, y_test, model_name):
    model.load_state_dict(torch.load(model_name))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test).float().to(device)
        predictions = model(inputs).squeeze().cpu().numpy()
    mse = mean_squared_error(y_test, predictions)
    return mse, predictions

# Evaluate each model
rnn_mse, rnn_predictions = evaluate_model(rnn_model, X_test, y_test, "best_rnn_model.pth")
lstm_mse, lstm_predictions = evaluate_model(lstm_model, X_test, y_test, "best_lstm_model.pth")
gru_mse, gru_predictions = evaluate_model(gru_model, X_test, y_test, "best_gru_model.pth")

print(f"RNN MSE: {rnn_mse:.4f}")
print(f"LSTM MSE: {lstm_mse:.4f}")
print(f"GRU MSE: {gru_mse:.4f}")

# Plotting the actual vs predicted values for each model
plt.figure(figsize=(15, 10))

# Plot RNN predictions vs actual values
plt.subplot(3, 1, 1)
plt.plot(y_test, label='Actual Values', color='blue')
plt.plot(rnn_predictions, label='RNN Predictions', color='red', linestyle='dashed')
plt.title('RNN - Actual vs Predicted')
plt.xlabel('Test Sample Index')
plt.ylabel('Predicted/Actual Values')
plt.legend()

# Plot LSTM predictions vs actual values
plt.subplot(3, 1, 2)
plt.plot(y_test, label='Actual Values', color='blue')
plt.plot(lstm_predictions, label='LSTM Predictions', color='green', linestyle='dashed')
plt.title('LSTM - Actual vs Predicted')
plt.xlabel('Test Sample Index')
plt.ylabel('Predicted/Actual Values')
plt.legend()

# Plot GRU predictions vs actual values
plt.subplot(3, 1, 3)
plt.plot(y_test, label='Actual Values', color='blue')
plt.plot(gru_predictions, label='GRU Predictions', color='orange', linestyle='dashed')
plt.title('GRU - Actual vs Predicted')
plt.xlabel('Test Sample Index')
plt.ylabel('Predicted/Actual Values')
plt.legend()

plt.tight_layout()
plt.show()

        

