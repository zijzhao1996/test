from tqdm import tqdm

# Modified training loop with progress bar and metric storage
train_losses = []
val_losses = []
train_ics = []
val_ics = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_outputs, train_targets_collected = [], []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
    for features, targets in pbar:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_outputs.extend(outputs.detach().cpu().numpy())
        train_targets_collected.extend(targets.detach().cpu().numpy())
        pbar.set_postfix({'loss': loss.item()})

    train_losses.append(train_loss / len(train_loader))
    train_ic = pearsonr(train_targets_collected, train_outputs)[0]
    train_ics.append(train_ic)

    # Validation
    model.eval()
    val_loss = 0.0
    val_outputs, val_targets_collected = [], []
    pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for features, targets in pbar:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_outputs.extend(outputs.detach().cpu().numpy())
            val_targets_collected.extend(targets.detach().cpu().numpy())
            pbar.set_postfix({'loss': loss.item()})

    val_losses.append(val_loss / len(val_loader))
    val_ic = pearsonr(val_targets_collected, val_outputs)[0]
    val_ics.append(val_ic)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train IC: {train_ic:.4f}, Val Loss: {val_losses[-1]:.4f}, Val IC: {val_ic:.4f}")

# ... Rest of the plotting code remains the same





import torch
from torch.utils.data import Dataset, DataLoader

class FinancialDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

        # Extract features and labels
        self.features = self.dataframe.filter(like='hist_ret').values
        self.labels = self.dataframe['target'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Convert to tensors
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label

# Create datasets
train_dataset = FinancialDataset(train_df)
valid_dataset = FinancialDataset(valid_df)

# Create dataloaders
batch_size = 64  # You can modify the batch size as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Now train_dataloader and valid_dataloader can be used in your training loop





import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from tqdm import tqdm

# Initialize the model
input_dim = train_df.filter(like='hist_ret').shape[1]  # Assuming all 'hist_ret' columns are features
model = Linear(input_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training settings
num_epochs = 10  # Example epoch count, adjust as necessary

# Training loop (as provided)
train_losses = []
val_losses = []
train_ics = []
val_ics = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_outputs, train_targets_collected = [], []
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
    for features, targets in pbar:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_outputs.extend(outputs.detach().cpu().numpy())
        train_targets_collected.extend(targets.detach().cpu().numpy())
        pbar.set_postfix({'loss': loss.item()})

    train_losses.append(train_loss / len(train_dataloader))
    train_ic = pearsonr(train_targets_collected, train_outputs)[0]
    train_ics.append(train_ic)

    # Validation
    model.eval()
    val_loss = 0.0
    val_outputs, val_targets_collected = [], []
    pbar = tqdm(valid_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for features, targets in pbar:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_outputs.extend(outputs.detach().cpu().numpy())
            val_targets_collected.extend(targets.detach().cpu().numpy())
            pbar.set_postfix({'loss': loss.item()})

    val_losses.append(val_loss / len(valid_dataloader))
    val_ic = pearsonr(val_targets_collected, val_outputs)[0]
    val_ics.append(val_ic)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train IC: {train_ic:.4f}, Val Loss: {val_losses[-1]:.4f}, Val IC: {val_ic:.4f}")

# Plotting or further analysis...

