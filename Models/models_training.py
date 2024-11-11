
# Training function
def train_model(model, X_train, y_train, X_test, y_test, epochs=50, lr=0.001, model_name="best_model.pth"):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Move data to device
        inputs = torch.tensor(X_train).float().to(device)
        targets = torch.tensor(y_train).float().to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_inputs = torch.tensor(X_test).float().to(device)
            val_targets = torch.tensor(y_test).float().to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs.squeeze(), val_targets)
            val_losses.append(val_loss.item())
        
        # Save the model if validation loss improves
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), model_name)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    return model, train_losses, val_losses

# Hyperparameters
input_size = X_train.shape[2]  # Set input size to the number of features
hidden_size = 128
output_size = 1
epochs = 100
learning_rate = 0.001

# Initialize models
rnn_model = RNNModel(input_size, hidden_size, output_size)
lstm_model = LSTMModel(input_size, hidden_size, output_size)
gru_model = GRUModel(input_size, hidden_size, output_size)

# Train each model and save the best model
print("Training RNN model...")
rnn_model, rnn_train_loss, rnn_val_loss = train_model(rnn_model, X_train, y_train, X_test, y_test, epochs, learning_rate, "best_rnn_model.pth")

print("
Training LSTM model...")
lstm_model, lstm_train_loss, lstm_val_loss = train_model(lstm_model, X_train, y_train, X_test, y_test, epochs, learning_rate, "best_lstm_model.pth")

print("
Training GRU model...")
gru_model, gru_train_loss, gru_val_loss = train_model(gru_model, X_train, y_train, X_test, y_test, epochs, learning_rate, "best_gru_model.pth")

