import os
import pandas as pd

input_file = r"C:\Projekte\chess_elo_prediction\data\chess_games.csv"
output_dir = "data/split_data"
os.makedirs(output_dir, exist_ok=True)
cols = ["WhiteElo", "BlackElo", "AN"]
chunksize = 100000
target = 1250000
current_file = 1
current_count = 0

for chunk in pd.read_csv(input_file, usecols=cols, chunksize=chunksize):
    start_idx = 0
    while start_idx < len(chunk) and current_file <= 5:
        remaining = target - current_count
        subset = chunk.iloc[start_idx:start_idx + remaining]
        out_path = os.path.join(output_dir, f"set_{current_file}.csv")
        subset.to_csv(out_path, mode='a', header=(current_count == 0), index=False)
        n_written = len(subset)
        current_count += n_written
        start_idx += n_written
        if current_count >= target:
            current_file += 1
            current_count = 0
            if current_file > 5:
                break