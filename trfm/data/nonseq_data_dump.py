import torch
from torch.utils.data import Dataset


class NoseqDataset(Dataset):
    def __init__(self, dataframe, scale=1, downsample=False):
        """
        Initialize the dataset.

        Args:
        dataframe (pd.DataFrame): Source data.
        scale (float): Scaling factor for the data.
        downsample (bool): If True, downsample the data.
        """
        self.dataframe = dataframe
        if downsample:
            self.dataframe = self.dataframe[self.dataframe['bar_time'].astype(str).str.endswith('0000')] #WARNING

        selected_cols = ['target'] + [col for col in self.dataframe.columns if col.startswith('hist_ret')]
        self.dataframe = self.dataframe[selected_cols].dropna()
        self.dataframe = self.dataframe.reset_index(drop=True)

        self.features = self.dataframe.filter(like='hist_ret').values * scale
        self.labels = self.dataframe['target'].values * scale

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Fetches the features and label for a given index."""
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label