import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        
class ChessModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, n_layers=2, dropout=0.1, nhead=4):
        super(ChessModel, self).__init__()
        self.use_transformer = use_transformer
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_rating = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)
        if self.use_transformer:
            x = self.positional_encoding(x)
            x = x.transpose(0, 1)
            x = self.transformer_encoder(x)
            x = x.transpose(0, 1)
            ratings = self.fc_rating(x)
        else:
            output, (h_n, c_n) = self.lstm(x)
            output = self.dropout(output)
            ratings = self.fc_rating(output)
        return ratings