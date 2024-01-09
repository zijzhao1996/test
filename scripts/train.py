import argparse
import logging
from trfm.trainer import Trainer
from trfm.data.dataloader import create_dataloader
from trfm.data.utils import determine_years

def main(config_path):
    # Load configuration
    trainer = Trainer(config_path)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

    # Extract test year from the configuration
    test_years = trainer.config['data_params']['test_year']
    assert isinstance(test_years, list), 'Years must be a list'
    is_seq = trainer.config['data_params']['is_seq']
    seq_len = trainer.config['data_params']['seq_len'] if is_seq else None

    # Determine training and validation years
    for test_year in test_years:
        train_years, valid_years = determine_years(test_year)
        logging.info(f"Satrt to test on year {test_year}...")
        # Create DataLoaders for training, validation, and testing
        train_dataloader = create_dataloader(years=train_years,
                                            batch_size=trainer.config['training_params']['batch_size'],
                                            shuffle=True,
                                            scale=1e4,
                                            seq_len=seq_len,
                                            downsample=True,
                                            is_seq=is_seq)
        valid_dataloader = create_dataloader(years=valid_years,
                                            batch_size=trainer.config['training_params']['batch_size'],
                                            shuffle=False,
                                            scale=1e4,
                                            seq_len=seq_len,
                                            downsample=True,
                                            is_seq=is_seq)
        test_dataloader = create_dataloader(years=[str(test_year)],
                                            batch_size=trainer.config['training_params']['batch_size'],
                                            shuffle=False,
                                            scale=1e4,
                                            seq_len=seq_len,
                                            downsample=True,
                                            is_seq=is_seq)

        # Train the model
        trainer.train(train_dataloader, valid_dataloader, test_dataloader, test_year)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    main(args.config)
