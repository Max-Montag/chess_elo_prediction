import torch.nn as nn

class ChessModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, n_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc_rating = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        h = h[-1]
        rating = self.fc_rating(h)
        return rating