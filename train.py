import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import wandb

wandb.init(project="chess_elo_prediction", config={
    "learning_rate": 1e-3,
    "batch_size": 32,
    "embed_dim": 64,
    "hidden_dim": 64,
    "n_layers": 1
})

df = pd.read_pickle("archive/games_train.pkl")
print("using ", len(df), " games for training")
vocab_size = max(max(seq) for seq in df["moves_encoded"]) + 1

class ChessModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, n_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc_rating = nn.Linear(hidden_dim, 2)
        self.fc_winner = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        h = h[-1]
        rating = self.fc_rating(h)
        winner = self.fc_winner(h)
        return rating, winner

sequences = [torch.tensor(seq, dtype=torch.long) for seq in df["moves_encoded"]]
X = pad_sequence(sequences, batch_first=True, padding_value=0)
ratings = torch.tensor(df[["black_rating_scaled", "white_rating_scaled"]].values, dtype=torch.float)
winner = torch.tensor(df["winner_encoded"].values, dtype=torch.float)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessModel(vocab_size=vocab_size,
                   embed_dim=wandb.config.embed_dim,
                   hidden_dim=wandb.config.hidden_dim,
                   n_layers=wandb.config.n_layers).to(device)

criterion_mse = nn.MSELoss()
criterion_bce = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

wandb.watch(model, log="all")

batch_indices = np.random.choice(len(X), size=wandb.config.batch_size, replace=False)
X_batch = X[batch_indices].to(device)
ratings_batch = ratings[batch_indices].to(device)
winner_batch = winner[batch_indices].to(device)

model.train()
optimizer.zero_grad()
pred_ratings, pred_winner = model(X_batch)
loss_ratings = criterion_mse(pred_ratings, ratings_batch)
loss_winner = criterion_bce(pred_winner.squeeze(1), winner_batch)
loss = loss_ratings + loss_winner
loss.backward()
optimizer.step()

torch.save(model.state_dict(), "model.pth")

wandb.log({
    "loss_ratings": loss_ratings.item(),
    "loss_winner": loss_winner.item(),
    "loss_total": loss.item()
})

wandb.finish()

print("Model saved!")
print("--- Results ---")
print("Loss ratings:", loss_ratings.item())
print("Loss winner:", loss_winner.item())
