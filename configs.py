config_a = {
    "name": "a",
    "seed": 16,
    "learning_rate": 0.000001,
    "dataset_size": 40000,
    "patience": 5,
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

config_b = {
    "name": "b",
    "seed": 16,
    "learning_rate": 0.000001,
    "dataset_size": 40000,
    "patience": 5,
    "dropout": 0.5,
    "num_epochs": 200,
    "batch_size": 64,
    "vocab_size": 11117,
    "embed_dim": 128,
    "hidden_dim": 128,
    "n_layers": 2,
    "mask_prob": 0.5,
    "mask_token": 11116,
    "weight_decay": 0.0001
}