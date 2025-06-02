import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Statistics for the era5_temp2m_16x_train dataset (in Kelvin)
T_MIN = 202.66
T_MAX = 320.93


class ERA5Dataset(Dataset):
    """ERA5 temperature dataset.

    Args:
        path_to_data (string): Path to directory where data is stored.
        transform (torchvision.Transform): Optional transform to apply to data.
        normalize (bool): Whether to normalize data to lie in [0, 1]. Defaults
            to True.
    """
    def __init__(self, path_to_data, transform=None, normalize=True):
        self.path_to_data = path_to_data
        self.transform = transform
        self.normalize = normalize
        self.filepaths = glob.glob(path_to_data + '/*.npz')
        self.filepaths.sort()  # Ensure consistent ordering of paths

    def __getitem__(self, index):
        # Dictionary containing latitude, longitude and temperature
        data = np.load(self.filepaths[index])
        latitude = data['latitude']  # Shape (num_lats,)
        longitude = data['longitude']  # Shape (num_lons,)
        temperature = data['temperature']  # Shape (num_lats, num_lons)
        if self.normalize:
            temperature = (temperature - T_MIN) / (T_MAX - T_MIN)
            # Move to [-1,1]
            temperature = 2. * temperature - 1.
            
        # Create a grid of latitude and longitude values matching the shape
        # of the temperature grid
        longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
        # Shape (3, num_lats, num_lons)
        data_tensor = np.stack([latitude_grid, longitude_grid, temperature])
        data_tensor = torch.Tensor(data_tensor)
        # Perform optional transform
        if self.transform:
            data_tensor = self.transform(data_tensor)
        return data_tensor, 0  # Label to ensure consistency with image datasets

    def __len__(self):
        return len(self.filepaths)