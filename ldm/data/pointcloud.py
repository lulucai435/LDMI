import numpy as np
import torch
import glob
import os
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    """3D point cloud dataset with optional missing data handling.
    
    Each datapoint: (num_points, 4) tensor with x,y,z coordinates and binary label.
    """
    def __init__(self, data_root, split="train", missing_perc=0.0):
        
        base_folder = "shapenet_point_clouds"
        data_root = os.path.join(data_root, base_folder)
        self.data = torch.load(os.path.join(data_root, f"tensor_{split}.pt"))
        self.missing_percent = missing_perc
        self.missing_data = missing_perc > 0

    def __getitem__(self, index):
        point_cloud = self.data[index]
        # Scale coordinates from [-.5, .5] to [-1, 1]
        point_cloud = torch.cat((2. * point_cloud[:,:3], point_cloud[:,[-1]]), dim=1)
        
        if not self.missing_data:
            return point_cloud, 0
            
        missing_rate = self._get_missing_rate()
        observed_mask = self._create_mask(point_cloud.shape, missing_rate)
        return point_cloud, 0, observed_mask

    def _get_missing_rate(self):
        if self.missing_percent == 1:
            return np.random.rand(1) * 0.9
        elif self.missing_percent > 0:
            return np.random.uniform(0, self.missing_percent)
        return -1

    def _create_mask(self, shape, rate):
        mask_1d = np.random.rand(shape[0]) > rate
        return np.repeat(mask_1d[:, np.newaxis], shape[1], axis=1)

    def __len__(self):
        return len(self.data)

class VoxelDataset(Dataset):
    """3D voxel dataset with split handling.
    
    Loads and processes voxel data with optional resizing.
    """
    def __init__(self, data_root, split='train', size=32, threshold=0.05, include_idx=False):
        
        base_folder = "shapenet_voxels"
        data_root = os.path.join(data_root, base_folder)
    
        self.voxel_paths = sorted(glob.glob(f"{data_root}/*.pt"))
        self.size = size
        self.threshold = threshold
        self.missing_data = True
        self.split = split
        self._setup_split(split)

    def _setup_split(self, split):
        num_samples = len(self.voxel_paths)
        num_train = int(0.8 * num_samples)
        num_val = int(0.1 * num_samples)
        
        self.split_sizes = {
            'train': num_train,#16, #DEBUG
            'val': num_val,
            'test': num_samples - (num_train + num_val)
        }
        
        self.start_index = {
            'train': 0,
            'val': num_train,
            'test': num_train + num_val
        }[split]

    def __getitem__(self, index):
        """
        Load and process voxel data.

        Returns: voxels (32,32,32), label (0 for all samples)
        
        """
        voxels = torch.load(self.voxel_paths[self.start_index + index])
        voxels = voxels.unsqueeze(0).float()
        
        if self.size != 32:
            voxels = self._resize_voxels(voxels)
        return voxels, 0

    def _resize_voxels(self, voxels):
        resized = torch.nn.functional.interpolate(
            voxels.unsqueeze(0), 
            self.size,
            mode='trilinear'
        )[0]
        return resized > self.threshold

    def __len__(self):
        return self.split_sizes[self.split]

    def random_indices(self, num_indices, max_idx):
        """Generate random indices without replacement."""
        return torch.randperm(max_idx)[:num_indices]
    


class PointCloudTrain(PointCloudDataset):
    def __init__(self, data_root, **kwargs):
        super().__init__(data_root, split="train", **kwargs)

class PointCloudVal(PointCloudDataset):
    def __init__(self, data_root, **kwargs):
        super().__init__(data_root, split="val", **kwargs)

class PointCloudTest(PointCloudDataset):
    def __init__(self, data_root, **kwargs):
        super().__init__(data_root, split="test", **kwargs)

class VoxelTrain(VoxelDataset):
    def __init__(self, data_root, **kwargs):
        super().__init__(data_root, split="train", **kwargs)

class VoxelVal(VoxelDataset):
    def __init__(self, data_root, **kwargs):
        super().__init__(data_root, split="val", **kwargs)

class VoxelTest(VoxelDataset):
    def __init__(self, data_root, **kwargs):
        super().__init__(data_root, split="test", **kwargs)