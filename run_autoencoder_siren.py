import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import yaml
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


class SingleImageDataset(Dataset):
    def __init__(self, image_path: str, image_size: int = 256):
        super().__init__()
        self.image_size = image_size

        # Load and transform image
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Convert to [-1,1]
            ]
        )

        self.image = Image.open(image_path).convert("RGB")
        self.image_tensor = transform(self.image).permute(1, 2, 0)

    def __len__(self):
        return int(1e8)  # Infinite iterations for single image

    def __getitem__(self, idx):
        return {"image": self.image_tensor.clone()}


def load_config(config_path):
    """Load config from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def main():
    # Configuration
    CONFIG_PATH = "configs/ivae/imagenet_ivae.yaml"
    TARGET_IMAGE_PATH = "/mnt/Files/workspace/HyperNerf/LDMI/data/imagenet100/train/n01440764/n01440764_438.JPEG"

    # Load config
    config = load_config(CONFIG_PATH)

    # Override certain config values for single image training
    config.model.params.encoder = None  # No encoder needed
    config.model.params.ckpt_path = None  # Don't load checkpoint
    config.model.params.lossconfig = None  # No complex loss config needed

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Create dataset and dataloader
    dataset = SingleImageDataset(
        TARGET_IMAGE_PATH, image_size=config.model.params.decoder.params.data_shape[0]
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize model from config
    model = instantiate_from_config(config.model)

    # Initialize logger
    logger = TensorBoardLogger("logs", name="autoencoder_siren")

    # Initialize trainer with callbacks
    trainer = pl.Trainer(
        max_steps=int(1e5),
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(logger.log_dir, "checkpoints"),
                filename="{epoch}-{train_loss:.4f}",
                monitor="train/loss",
                save_top_k=3,
                mode="min",
            ),
            TQDMProgressBar(refresh_rate=20),
        ],
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    # Train model
    trainer.fit(model, dataloader)
    print("\nTraining completed! To view results, run:")
    print("tensorboard --logdir=logs")
    print("Then open http://localhost:6006 in your browser")


if __name__ == "__main__":
    main()
