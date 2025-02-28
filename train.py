import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import wandb

wandb.init(project="chess_elo_prediction", config={
    "learning_rate": 1e-2,
    "num_epochs": 10,
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
        #self.fc_winner = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        h = h[-1]
        rating = self.fc_rating(h)
        #winner = self.fc_winner(h)
        return rating#, winner

print("vocab_size:", vocab_size)
print("lencoded:", len(df["moves_encoded"]))
sequences = [torch.tensor(seq, dtype=torch.long) for seq in df["moves_encoded"]]
X = pad_sequence(sequences, batch_first=True, padding_value=0)
print("X.shape:", X.shape)
ratings = torch.tensor(df[["black_rating_scaled", "white_rating_scaled"]].values, dtype=torch.float)
#winner = torch.tensor(df["winner_encoded"].values, dtype=torch.float)
print("ratings.shape:", ratings.shape)
#print("winner.shape:", winner.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessModel(vocab_size=vocab_size,
                   embed_dim=wandb.config.embed_dim,
                   hidden_dim=wandb.config.hidden_dim,
                   n_layers=wandb.config.n_layers).to(device)

criterion_mse = nn.MSELoss()
criterion_bce = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

wandb.watch(model, log="all")

dataset = TensorDataset(X, ratings)#, winner)
data_loader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True)

# train loop
for epoch in range(wandb.config.num_epochs):
    loss = 0.0
    for X_batch, ratings_batch in data_loader: #, winner_batch in data_loader:
        X_batch = X_batch.to(device)
        ratings_batch = ratings_batch.to(device)
        #winner_batch = winner_batch.to(device)
        model.train()
        optimizer.zero_grad()
        #pred_ratings, pred_winner = model(X_batch)#
        pred_ratings = model(X_batch)
        loss_ratings = criterion_mse(pred_ratings, ratings_batch)
        #loss_winner = criterion_bce(pred_winner.squeeze(1), winner_batch)
        new_loss = loss_ratings # + loss_winner
        new_loss.backward()
        optimizer.step()
        loss += new_loss.item()
    avg_loss = loss / len(data_loader)
    wandb.log({"epoch": epoch, "loss_total": avg_loss})

# save model
i = 0
while os.path.exists(f"models/model_{i}.pth"):
    i += 1
torch.save(model.state_dict(), f"models/model_{i}.pth")

# log results
wandb.log({
    "loss_ratings": loss_ratings.item(),
    #"loss_winner": loss_winner.item(),
    "loss_total": new_loss.item()
})

wandb.finish()

print("Model saved!")
print("--- Results ---")
print("Loss ratings:", loss_ratings.item())
#print("Loss winner:", loss_winner.item())
