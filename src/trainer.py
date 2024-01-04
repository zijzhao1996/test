import torch
import yaml
import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.stats import pearsonr
from src.model.linear import Linear  # Adjust the import path as necessary

class Trainer:
    def __init__(self, config_path):
        # Load hyperparameters and model configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Initialize model, criterion, and optimizer
        self.model = Linear(input_dim=self.config['model_params']['input_dim'])
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['training_params']['learning_rate'])

        # Tensorboard writer
        self.writer = SummaryWriter()

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(self, dataloader):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        total_samples = len(dataloader.dataset)
        predictions, targets = [], []

        for features, target in tqdm(dataloader, desc="Training", leave=False):
            features, target = features.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(features)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * features.size(0)
            predictions.extend(output.detach().cpu().numpy())
            targets.extend(target.detach().cpu().numpy())

        avg_loss = total_loss / total_samples
        ic = pearsonr(targets, predictions)[0]
        return avg_loss, ic

    def validate_epoch(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_samples = len(dataloader.dataset)
        predictions, targets = [], []

        with torch.no_grad():
            for features, target in tqdm(dataloader, desc="Validation", leave=False):
                features, target = features.to(self.device), target.to(self.device)
                output = self.model(features)
                loss = self.criterion(output, target)

                total_loss += loss.item() * features.size(0)
                predictions.extend(output.detach().cpu().numpy())
                targets.extend(target.detach().cpu().numpy())

        avg_loss = total_loss / total_samples
        ic = pearsonr(targets, predictions)[0]
        return avg_loss, ic

    def train(self, train_dataloader, valid_dataloader):
        """Train the model across all epochs."""
        for epoch in range(self.config['training_params']['num_epochs']):
            train_loss, train_ic = self.train_epoch(train_dataloader)
            val_loss, val_ic = self.validate_epoch(valid_dataloader)

            # Log metrics
            self.writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
            self.writer.add_scalars('IC', {'Train': train_ic, 'Validation': val_ic}, epoch)
            logging.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train IC: {train_ic:.4f}, Val Loss: {val_loss:.4f}, Val IC: {val_ic:.4f}")

        # Close the Tensorboard writer
        self.writer.close()

# Example usage is demonstrated in the main training script `scripts/train.py`.
