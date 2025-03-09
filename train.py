import os
import ast
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import wandb
from chess_model import ChessModel
from configs import *
from masking import mask_batch

current_config = config_b
wandb.init(project="chess-rating-prediction", config=current_config)

data_full = pd.read_csv("data/split_data_prepared/set_1_normalized.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessModel(vocab_size=wandb.config.vocab_size,
                   embed_dim=wandb.config.embed_dim,
                #    hidden_dim=wandb.config.hidden_dim,
                   n_layers=wandb.config.n_layers,
                   dropout=wandb.config.dropout,
                   nhead=wandb.config.nhead,
                   ).to(device)
# wandb.watch(model, log="all")
criterion_mse = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)

df_train, df_val = train_test_split(data_full, test_size=0.004, random_state=wandb.config.seed)  # TODO USE SET 5 FOR TESTING!

sequences_val = [torch.tensor(ast.literal_eval(seq), dtype=torch.long) for seq in df_val["moves_encoded"]]
X_val = pad_sequence(sequences_val, batch_first=True, padding_value=0)
ratings_val = torch.tensor(df_val["white_rating_scaled"].values, dtype=torch.float).unsqueeze(1)

val_dataset = TensorDataset(X_val, ratings_val)
val_data_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=True)

def reload_data(data, seed):
    print("Reloading data with seed ", seed)
    sample = data.sample(n=wandb.config.dataset_size, random_state=seed).reset_index(drop=True)

    sequences = [torch.tensor(ast.literal_eval(seq), dtype=torch.long) for seq in sample["moves_encoded"]]
    X = pad_sequence(sequences, batch_first=True, padding_value=0)
    ratings = torch.tensor(sample["white_rating_scaled"].values, dtype=torch.float).unsqueeze(1)

    dataset = TensorDataset(X, ratings)
    data_loader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True)

    return data_loader

data_loader = reload_data(df_train, wandb.config.seed)

best_val_loss = float('inf')
reload_interval = wandb.config.reload_interval

# train loop
for epoch in range(wandb.config.num_epochs):
    if epoch % reload_interval == 0 and epoch != 0:
        data_loader = reload_data(df_train, wandb.config.seed + epoch)

    loss = 0.0
    for X_batch, ratings_batch in data_loader:
        X_batch = mask_batch(X_batch, wandb.config.mask_prob, wandb.config.mask_token)
        X_batch = X_batch.to(device)
        ratings_batch = ratings_batch.to(device)

        model.train()
        optimizer.zero_grad()
        pred_ratings = model(X_batch)

        seq_len = pred_ratings.shape[1]
        # weights = torch.linspace(0.1, 1.0, steps=seq_len).to(device) # higher weight for later predictions!
        # weights = weights.unsqueeze(0).unsqueeze(-1)

        loss = criterion_mse(pred_ratings, ratings_batch.unsqueeze(1).expand_as(pred_ratings))
        # weighted_loss = (loss * weights).mean()
        # weighted_loss.backward()
        loss.backward()
        optimizer.step()
        # loss += weighted_loss.item()
        loss += loss.item()
    avg_loss = loss / len(data_loader)
    wandb.log({"loss_total": avg_loss})

    # validation
    loss_val = 0.0
    percentage_error_sum = 0
    for X_batch, ratings_batch in val_data_loader:
        X_batch = X_batch.to(device)
        ratings_batch = ratings_batch.to(device)
        model.eval()
        pred_ratings_seq = model(X_batch)
        pred_ratings = pred_ratings_seq[:, -1, :]
        loss = criterion_mse(pred_ratings, ratings_batch)
        loss_val += loss.item()
        percentage_error_sum += torch.sum(torch.abs(pred_ratings - ratings_batch) / torch.abs(ratings_batch) * 100).item()
    avg_percentage_error = percentage_error_sum / ratings_val.numel()
    avg_val_loss = loss_val / len(val_data_loader)
    wandb.log({"loss_val": avg_val_loss, "percentage_error": avg_percentage_error})
    print(f"Epoch {epoch} - loss: {avg_loss*10000:.2f}, val_loss: {avg_val_loss*10000:.2f}, percentage_error: {avg_percentage_error:.2f}")

# save model
i = 0
while os.path.exists(f"models/model_{wandb.config.name}_{i}.pth"):
    i += 1
torch.save(model.state_dict(), f"models/model_{wandb.config.name}_{i}.pth")
print(f"Model saved as model_{wandb.config.name}_{i}.pth")

wandb.finish()
