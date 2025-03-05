import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import wandb
from chess_model import ChessModel
from configs import *
from masking import mask_batch

current_config = config_a
wandb.init(project="chess-rating-prediction", config=current_config)

data = pd.read_pickle("data/games_prepared.pkl")

df, temp_df = train_test_split(data, test_size=0.2, random_state=wandb.config.seed)
test_df, df_val = train_test_split(temp_df, test_size=0.2, random_state=wandb.config.seed)

test_df.to_pickle("data/games_test.pkl")

print(f"Train set size: {len(df)}")
print(f"Test set size: {len(test_df)}")
print(f"Validation set size: {len(df_val)}")

sequences = [torch.tensor(seq, dtype=torch.long) for seq in df["moves_encoded"]]
X = pad_sequence(sequences, batch_first=True, padding_value=0)
ratings = torch.tensor(df[["black_rating_scaled", "white_rating_scaled"]].values, dtype=torch.float)

sequences_val = [torch.tensor(seq, dtype=torch.long) for seq in df_val["moves_encoded"]]
X_val = pad_sequence(sequences_val, batch_first=True, padding_value=0)
ratings_val = torch.tensor(df_val[["black_rating_scaled", "white_rating_scaled"]].values, dtype=torch.float)

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

val_dataset = TensorDataset(X_val, ratings_val)
val_data_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=True)

for epoch in range(wandb.config.num_epochs):
    loss = 0.0
    for X_batch, ratings_batch in data_loader:
        X_batch = mask_batch(X_batch, wandb.config.mask_prob, wandb.config.mask_token)
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
    loss_val = 0.0
    percentage_error_sum = 0
    for X_batch, ratings_batch in val_data_loader:
        X_batch = X_batch.to(device)
        ratings_batch = ratings_batch.to(device)
        model.eval()
        pred_ratings = model(X_batch)
        new_loss = criterion_mse(pred_ratings, ratings_batch)
        loss_val += new_loss.item()
        percentage_error_sum += torch.sum(torch.abs(pred_ratings - ratings_batch) / torch.abs(ratings_batch) * 100).item()
    avg_percentage_error = percentage_error_sum / ratings_val.numel()
    wandb.log({"loss_val": loss_val / len(val_data_loader), "percentage_error": avg_percentage_error})
    print(f"Epoch {epoch} - loss: {avg_loss}, val_loss: {loss_val / len(val_data_loader)}, percentage_error: {avg_percentage_error}")

i = 0
while os.path.exists(f"models/model_{wandb.config.name}_{i}.pth"):
    i += 1
torch.save(model.state_dict(), f"models/model_{wandb.config.name}_{i}.pth")
print(f"Model saved as model_{wandb.config.name}_{i}.pth")

wandb.finish()
