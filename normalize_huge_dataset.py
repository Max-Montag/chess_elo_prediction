import re
import pandas as pd

input_file = r"C:\Projekte\chess_elo_prediction\data\split_data\set_1.csv"
output_file = r"C:\Projekte\chess_elo_prediction\data\split_data_prepared\set_1_normalized.csv"
cols = ["WhiteElo", "BlackElo", "AN"]

unique_moves = set()

def is_valid_row(token_list):
    for token in token_list:
        if token in {"{", "}"} or "[%eval" in token:
            return False
    return True

df = pd.read_csv(input_file, usecols=cols)
max_rating = max(df["WhiteElo"].max(), df["BlackElo"].max())

df["AN"] = df["AN"].apply(lambda x: re.sub(r"\d+\.", "", x))
df["AN"] = df["AN"].str.replace("1-0", "", regex=False)
df["AN"] = df["AN"].str.replace("0-1", "", regex=False)
df["AN"] = df["AN"].str.replace("1/2-1/2", "", regex=False)
df["moves_tokenized"] = df["AN"].apply(lambda x: x.split())

df = df[df["moves_tokenized"].apply(is_valid_row)]
for token_list in df["moves_tokenized"]:
    unique_moves.update(token_list)

unique_moves = sorted(unique_moves)
move_to_idx = {move: i+1 for i, move in enumerate(unique_moves)}

df["black_rating_scaled"] = df["BlackElo"] / max_rating
df["white_rating_scaled"] = df["WhiteElo"] / max_rating
df["moves_encoded"] = df["moves_tokenized"].apply(lambda row: [move_to_idx[move] for move in row])
print(df.head())
df.drop(columns=["WhiteElo", "BlackElo", "AN", "moves_tokenized"], inplace=True)
print(df.head())
print("unique_moves", len(unique_moves))

# save as csv
df.to_csv(output_file, index=False)