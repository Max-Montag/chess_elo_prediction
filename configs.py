config_a = { # lstm
    "name": "a",
    "seed": 16,
    "learning_rate": 0.000001,
    "dataset_size": 40000,
    "reload_interval": 5,
    "dropout": 0.5,
    "num_epochs": 50,
    "batch_size": 64,
    "vocab_size": 11117,
    "embed_dim": 64,
    "hidden_dim": 64,
    "n_layers": 1,
    "mask_prob": 0.5,
    "mask_token": 11116
}

# config_b = { # transformer
#     "name": "b",
#     "seed": 17,
#     "learning_rate": 0.0000004,
#     "dataset_size": 80000,
#     "reload_interval": 30,
#     "dropout": 0.47,
#     "num_epochs": 30,
#     "batch_size": 64,
#     "vocab_size": 11117,
#     "embed_dim": 64,
#     "hidden_dim": 128,
#     "n_layers": 2,
#     "mask_prob": 0.25,
#     "mask_token": 11116,
#     "weight_decay": 0.00013,
#     "nhead": 4
# }

config_b = { # transformer
    "name": "b",
    "seed": 17,
    "learning_rate": 0.0000005,
    "dataset_size": 50000,
    "reload_interval": 5,
    "dropout": 0.45,
    "num_epochs": 150,
    "batch_size": 64,
    "vocab_size": 11117,
    "embed_dim": 128,
    # "hidden_dim": 128,
    "n_layers": 2,
    "mask_prob": 0.35,
    "mask_token": 11116,
    "weight_decay": 0.00015,
    "nhead": 8
}
