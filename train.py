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
from masking import mask_batch

rating_threshold = 50 # 50 Elo points (Lichess rating scale) (only for test set evaluation)
max_rating = 3110

# import yaml'
# with open("configs/config_b.yaml", "r") as f:
#     current_config = yaml.safe_load(f)
# wandb.init(project="chess-rating-prediction", config=current_config)

wandb.init(project="chess-rating-prediction")

data_full = pd.read_csv("data/final_normalized.csv")

# balance dataset
data_full["rating_bin"] = pd.cut(data_full["white_rating_scaled"], bins=wandb.config.bins, labels=False)
groups = data_full.groupby("rating_bin")
min_count = groups.size().min()

df_bal_subset = groups.apply(lambda x: x.sample(min_count, random_state=wandb.config.seed)).reset_index(drop=True)
df_bal_subset = df_bal_subset.sample(frac=1, random_state=wandb.config.seed).reset_index(drop=True)
df_bal_subset.drop(columns=["rating_bin"], inplace=True)
print("balanced subset info", df_bal_subset.info())

# use everything else as test set
df_test = data_full[~data_full.index.isin(df_bal_subset.index)]
df_test.reset_index(drop=True, inplace=True)
df_test.drop(columns=["rating_bin"], inplace=True)
print("test set info", df_test.info())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessModel(vocab_size=wandb.config.vocab_size,
                   embed_dim=wandb.config.embed_dim,
                   n_layers=wandb.config.n_layers,
                   dropout=wandb.config.dropout,
                   nhead=wandb.config.nhead,
                   activation=wandb.config.activation
                   ).to(device)
# wandb.watch(model, log="all")

def get_criterion():
    if wandb.config.criterion == "MSE":
        return nn.MSELoss()
    elif wandb.config.criterion == "L1":
        return nn.L1Loss()
    elif wandb.config.criterion == "Smooth_L1":
        return nn.SmoothL1Loss()
    elif wandb.config.criterion == "Huber":
        return nn.HuberLoss(reduction='mean', delta=1.0)

criterion = get_criterion()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)

df_train, df_val = train_test_split(data_full, test_size=0.2, random_state=wandb.config.seed)

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
best_model_state = None
early_stopped = False
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
        loss = criterion(pred_ratings, ratings_batch.unsqueeze(1).expand_as(pred_ratings))
        if wandb.config.use_weighted_loss:
            weights = torch.linspace(0.1, 1.0, steps=seq_len).to(device)  # higher weight for later predictions!
            weights = weights.unsqueeze(0).unsqueeze(-1)
            weighted_loss = (loss * weights).mean()
            weighted_loss.backward()
        else:
            loss.backward()
        optimizer.step()
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
        loss = criterion(pred_ratings, ratings_batch)
        loss_val += loss.item()
        percentage_error_sum += torch.sum(torch.abs(pred_ratings - ratings_batch) / torch.abs(ratings_batch) * 100).item()
    avg_percentage_error = percentage_error_sum / ratings_val.numel()
    avg_val_loss = loss_val / len(val_data_loader)
    
    wandb.log({"loss_val": avg_val_loss, "percentage_error": avg_percentage_error})
    print(f"Epoch {epoch} - loss: {avg_loss*10000:.2f}, val_loss: {avg_val_loss*10000:.2f}, percentage_error: {avg_percentage_error:.2f}")
    
    # early stopping: no improvement -> save best model -> break
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
    else:
        print(f"Early stopping triggered at epoch {epoch} (val_loss did not improve)")
        torch.save(best_model_state, f"models/model_{wandb.config.name}_earlystop.pth")
        print(f"Model saved as model_{wandb.config.name}_earlystop.pth")
        early_stopped = True
        break

# save model
if not early_stopped:
    i = 0
    while os.path.exists(f"models/model_{wandb.config.name}_{i}.pth"):
        i += 1
    torch.save(model.state_dict(), f"models/model_{wandb.config.name}_{i}.pth")
    print(f"Model saved as model_{wandb.config.name}_{i}.pth")

# run evaluation on test set
model.load_state_dict(best_model_state)
model.eval()
sequences_test = [torch.tensor(ast.literal_eval(seq), dtype=torch.long) for seq in df_test["moves_encoded"]]
X_test = pad_sequence(sequences_test, batch_first=True, padding_value=0)
ratings_test = torch.tensor(df_test["white_rating_scaled"].values, dtype=torch.float).unsqueeze(1)

test_dataset = TensorDataset(X_test, ratings_test)
test_data_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=True)

loss_test = 0.0
percentage_error_sum = 0
accuracy_sum = 0
for X_batch, ratings_batch in test_data_loader:
    X_batch = X_batch.to(device)
    ratings_batch = ratings_batch.to(device)
    pred_ratings_seq = model(X_batch)
    pred_ratings = pred_ratings_seq[:, -1, :]
    loss = criterion(pred_ratings, ratings_batch)
    loss_test += loss.item()
    percentage_error_sum += torch.sum(torch.abs(pred_ratings - ratings_batch) / torch.abs(ratings_batch) * 100).item()
    correct = torch.abs(pred_ratings - ratings_batch) * max_rating < rating_threshold
    accuracy_sum = torch.sum(correct).item() / ratings_batch.numel()
avg_percentage_error = percentage_error_sum / ratings_test.numel()
avg_accuracy = accuracy_sum / len(test_data_loader)
avg_test_loss = loss_test / len(test_data_loader)
wandb.log({"loss_test": avg_test_loss, "percentage_error_test": avg_percentage_error, "accuracy_test": avg_accuracy})
print(f"Test loss: {avg_test_loss*10000:.2f}, percentage_error: {avg_percentage_error:.2f}, accuracy: {avg_accuracy:.2f}")

wandb.finish()
