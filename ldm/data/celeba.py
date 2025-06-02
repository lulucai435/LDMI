from torchvision.datasets import CelebA
import torchvision.transforms as transforms


class MyCelebA(CelebA):
    def __init__(self, data_root, split = "train", target_type = "attr", 
                 transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]), 
                 target_transform = None, download = False, **kwargs):
        super().__init__(data_root, split, target_type, transform, target_transform, download)

    def _check_integrity(self) -> bool:
        return True


class CelebaTrain(MyCelebA):
    def __init__(self, data_root, split = "train", **kwargs):
        super().__init__(data_root, split, **kwargs)
        

class CelebaVal(MyCelebA):
    def __init__(self, data_root, split = "valid",  **kwargs):
        super().__init__(data_root, split,  **kwargs)

class CelebaTest(MyCelebA):
    def __init__(self, data_root, split = "test", **kwargs):
        super().__init__(data_root, split, **kwargs)