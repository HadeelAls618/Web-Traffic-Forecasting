
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, output_size=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.batch_norm(out[:, -1, :])  # Use last timestep output
        out = self.dropout(out)
        out = self.fc(out)
        return out

