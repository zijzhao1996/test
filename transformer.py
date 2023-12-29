import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch import nn
from torch.nn import functional as F
from scipy.stats import pearsonr

# Data loading and preprocessing
def load_and_preprocess(years, scale=10000):
    tensors = []
    for year in years:
        file_path = f'{year}_data.parquet'
        df = pd.read_parquet(file_path)
        features = df[[col for col in df.columns if col.startswith('hist_return')]] * scale
        target = df['target'] * scale
        features_tensor = torch.tensor(features.values, dtype=torch.float32).unsqueeze(1)
        target_tensor = torch.tensor(target.values, dtype=torch.float32)
        tensors.append((features_tensor, target_tensor))
    features = torch.cat([t[0] for t in tensors], dim=0)
    targets = torch.cat([t[1] for t in tensors], dim=0)
    return features, targets

def create_dataloader(features, targets, batch_size=32, shuffle=True):
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Load data
TRAIN_YEARS = ['2008', '2009']
VALID_YEARS = ['2010']
train_features, train_targets = load_and_preprocess(TRAIN_YEARS)
val_features, val_targets = load_and_preprocess(VALID_YEARS)

# Create DataLoaders
train_loader = create_dataloader(train_features, train_targets)
val_loader = create_dataloader(val_features, val_targets, shuffle=False)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]





class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, T, F], [512, 60, 6]
        src = self.feature_layer(src)  # [512, 60, 8]

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()


# Instantiate the model, loss function, optimizer, etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(d_feat=train_features.shape[2], device=device).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)





def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for features, targets in data_loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_targets.extend(targets.tolist())
            all_outputs.extend(outputs.tolist())
    pearson_corr = pearsonr(all_targets, all_outputs)[0]
    return total_loss / len(data_loader), pearson_corr

# Training loop
import matplotlib.pyplot as plt

# Training loop with metric storage
train_losses = []
val_losses = []
train_ics = []
val_ics = []

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_corr = evaluate(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_ics.append(val_corr)

    # Evaluate training IC
    train_ic, _ = evaluate(model, train_loader, criterion, device)
    train_ics.append(train_ic)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Train IC: {train_ic}, Val IC: {val_corr}')

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Training and Validation Metrics')

# Train Loss
axs[0, 0].plot(train_losses, label='Train Loss')
axs[0, 0].set_title('Training Loss')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

# Validation Loss
axs[0, 1].plot(val_losses, label='Validation Loss', color='orange')
axs[0, 1].set_title('Validation Loss')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()

# Train IC
axs[1, 0].plot(train_ics, label='Train IC')
axs[1, 0].set_title('Training Information Coefficient')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('IC')
axs[1, 0].legend()

# Validation IC
axs[1, 1].plot(val_ics, label='Validation IC', color='green')
axs[1, 1].set_title('Validation Information Coefficient')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('IC')
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



