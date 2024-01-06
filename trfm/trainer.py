import torch
import yaml
import logging
import os
import importlib
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.stats import pearsonr
from .loss import get_loss_function
from .optim import get_optimizer
from .scheduler import get_scheduler

class Trainer:
    def __init__(self, config_path, log_dir='../logs'):
        """
        Initialize the Trainer class with model, loss, optimizer, and other training configurations.

        Args:
        config_path (str): Path to the configuration YAML file.
        log_dir (str): Directory for Tensorboard logs.
        """
        # Extract experiment name from config file name for logging purposes
        experiment_name = os.path.basename(config_path).split('.')[0]

        # Load hyperparameters and model configuration from YAML file
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Dynamically load and initialize the model from the configuration
        model_module = importlib.import_module(f"trfm.model.{self.config['model_params']['name']}")
        model_class = getattr(model_module, self.config['model_params']['name'])
        self.model = model_class(**{k: v for k, v in self.config['model_params'].items() if k != 'name'})

        # Initialize the loss function and optimizer with possible L2 regularization
        self.criterion = get_loss_function(self.config['loss'])
        weight_decay = self.config.get('training_params').get('weight_decay', 0)
        self.optimizer = get_optimizer(self.config['optimizer'], self.model.parameters(),
                                       self.config['training_params']['learning_rate'], weight_decay=weight_decay)

        # Tensorboard setup for monitoring training progress
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

        # Setup device for training (GPU or CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Initialize the learning rate scheduler if specified in config
        if 'scheduler' in self.config:
            scheduler_config = self.config['scheduler']
            scheduler_name = scheduler_config.pop('name')  # Extract the name of the scheduler
            self.scheduler = get_scheduler(self.optimizer, scheduler_name, **scheduler_config)
        else:
            self.scheduler = None

        # Setup for early stopping and model checkpointing
        self.best_val_loss = float('inf')
        self.patience = self.config.get('early_stopping_patience', 10)
        self.patience_counter = 0
        self.checkpoint_interval = self.config.get('checkpoint_interval', 5)
        self.checkpoint_dir = os.path.join(os.path.dirname(log_dir), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @staticmethod
    def set_seed(seed_value=42):
        """
        Set the random seed for reproducibility across runs.

        Args:
        seed_value (int): The seed value for random number generators.
        """
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
            # Add gradient clipping just before optimizer.step()
            clip_value = self.config.get('gradient_clipping', None)
            if clip_value:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
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

    def early_stopping_check(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True  # Early stopping triggered
        return False

    def save_checkpoint(self, epoch):
        if epoch % self.checkpoint_interval == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(self.model.state_dict(), checkpoint_path)

    def train(self, train_dataloader, valid_dataloader):
        """Train the model across all epochs."""
        self.set_seed()  # Set seed at the start of training
        for epoch in range(self.config['training_params']['num_epochs']):
            train_loss, train_ic = self.train_epoch(train_dataloader)
            val_loss, val_ic = self.validate_epoch(valid_dataloader)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Early stopping check
            if self.early_stopping_check(val_loss):
                logging.info("Early stopping triggered.")
                break

            # Save checkpoint
            self.save_checkpoint(epoch)

            # Log training metrics
            self.writer.add_scalar('Training Loss', train_loss, epoch)
            self.writer.add_scalar('Training IC', train_ic, epoch)

            # Log validation metrics
            self.writer.add_scalar('Validation Loss', val_loss, epoch)
            self.writer.add_scalar('Validation IC', val_ic, epoch)

            # Log information to the console
            logging.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train IC: {train_ic:.4f}, Val Loss: {val_loss:.4f}, Val IC: {val_ic:.4f}")

        # Close the Tensorboard writer
        self.writer.close()
