import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import wandb
from chess_model import ChessModel
from configs import *

current_config = config_a

wandb.init(project="chess-rating-prediction", config=current_config)

df = pd.read_pickle("data/games_train.pkl")
print("using ", len(df), " games for training")

sequences = [torch.tensor(seq, dtype=torch.long) for seq in df["moves_encoded"]]
X = pad_sequence(sequences, batch_first=True, padding_value=0)
ratings = torch.tensor(df[["black_rating_scaled", "white_rating_scaled"]].values, dtype=torch.float)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessModel(vocab_size=wandb.config.vocab_size,
                   embed_dim=wandb.config.embed_dim,
                   hidden_dim=wandb.config.hidden_dim,
                   n_layers=wandb.config.n_layers).to(device)

criterion_mse = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

wandb.watch(model, log="all")

dataset = TensorDataset(X, ratings)
data_loader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True)

# train loop
for epoch in range(wandb.config.num_epochs):
    loss = 0.0
    for X_batch, ratings_batch in data_loader:
        X_batch = X_batch.to(device)
        ratings_batch = ratings_batch.to(device)
        model.train()
        optimizer.zero_grad()
        pred_ratings = model(X_batch)
        new_loss = criterion_mse(pred_ratings, ratings_batch)
        new_loss.backward()
        optimizer.step()
        loss += new_loss.item()
    avg_loss = loss / len(data_loader)
    wandb.log({"loss_total": avg_loss})

# save model
i = 0
while os.path.exists(f"models/model_{current_config.name}_{i}.pth"):
    i += 1
torch.save(model.state_dict(), f"models/model_{current_config.name}_{i}.pth")
print (f"Model saved as model_{current_config.name}_{i}.pth")

wandb.finish()