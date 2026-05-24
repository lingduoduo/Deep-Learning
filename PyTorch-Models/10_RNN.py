import torch
import torch.nn as nn
import torch.optim as optim

seq_len = 10
n_samples = 100
X = torch.linspace(0, 4 * 3.14159, steps=n_samples).unsqueeze(1)
y = torch.sin(X)

def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i : i + seq_length])
        out_seq.append(data[i + seq_length])
    return torch.stack(in_seq), torch.stack(out_seq)

X_seq, y_seq = create_in_out_sequences(y, seq_length=seq_len)

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,            
        )
        self.predictor = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"Expected input with shape [batch, seq_len, input_size], got {tuple(x.shape)}")
        output, _ = self.rnn(x)
        last_step = output[:, -1, :]
        return self.predictor(last_step)

model = RNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 10
for epoch in range(epochs):
    for sequences, labels in zip(X_seq, y_seq):
        optimizer.zero_grad()

        sequences = sequences.unsqueeze(0)
        labels = labels.unsqueeze(0)

        predictions = model(sequences)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")