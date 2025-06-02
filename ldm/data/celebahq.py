from torchvision import datasets, transforms
from torch.utils.data import Dataset


class CelebAHQ(Dataset):
    def __init__(self, data_root, size=256):
        """
        A class to load the CelebAHQ dataset with predefined transformations.
        
        Args:
            path_to_data (str): Path to the dataset directory.
            size (int, optional): Image size to resize to. Default is 256.
        """
        self.data_root = data_root
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.dataset = datasets.ImageFolder(self.data_root, transform=self.transform)
    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Returns the image and label at the given index."""
        return self.dataset[idx]
