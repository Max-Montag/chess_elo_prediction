import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import wandb
from chess_model import ChessModel
from configs import *

use_config = config_a
use_model_no = None # highest index will be used if None

wandb.init(project="chess-rating-prediction-TEST", config=use_config)

df = pd.read_pickle("data/games_test.pkl")
print("using ", len(df), " games for testing")

sequences = [torch.tensor(seq, dtype=torch.long) for seq in df["moves_encoded"]]
X = pad_sequence(sequences, batch_first=True, padding_value=0)
ratings = torch.tensor(df[["black_rating_scaled", "white_rating_scaled"]].values, dtype=torch.float)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChessModel(vocab_size=use_config["vocab_size"],
                    embed_dim=use_config["embed_dim"],
                    hidden_dim=use_config["hidden_dim"],
                    n_layers=use_config["n_layers"]).to(device)

criterion_mse = nn.MSELoss()

# load model (by use_model_no or highest index)
if use_model_no is None:
    candidates = []
    for file in os.listdir("models"):
        if file.startswith("model_" + use_config["name"]):
            candidates.append(file)
    if len(candidates) > 0:
        candidates.sort()
        model.load_state_dict(torch.load("models/" + candidates[-1]))
    else:
        print("No model found for this config!")
        exit()
else:
    model.load_state_dict(torch.load(f"models/model_{use_config['name']}_{use_model_no}.pth"))

# forward test
model.eval()

dataset = TensorDataset(X, ratings)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

loss = 0.0
accuracy = 0
for X_batch, ratings_batch in data_loader:
    X_batch = X_batch.to(device)
    ratings_batch = ratings_batch.to(device)
    pred_ratings = model(X_batch)
    new_loss = criterion_mse(pred_ratings, ratings_batch)
    loss += new_loss.item()
    accuracy += torch.sum(torch.abs(pred_ratings - ratings_batch) < 100).item()
avg_loss = loss / len(data_loader)
accuracy = accuracy / len(df)
wandb.log({"loss_total": avg_loss, "accuracy": accuracy})
print("Accuracy: ", accuracy)
print("Average loss: ", avg_loss)

wandb.finish()
