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
