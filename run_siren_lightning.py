import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Any, Dict
import math
from datetime import datetime
import os
from PIL import Image
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader


class SIRENDataset(Dataset):
    def __init__(self, image_size: int, target_image_path: Optional[str] = None):
        super().__init__()
        self.image_size = image_size
        self.device = torch.device("cpu")  # Data prep on CPU

        # Generate coordinates
        self.coords = self._generate_coordinates()

        # Generate or load target
        if target_image_path and os.path.exists(target_image_path):
            self.target = self._load_target_image(target_image_path)
            print(f"Loaded target image from: {target_image_path}")
        else:
            self.target = self._generate_target_image()
            print("Using generated pattern as target")

    def _generate_coordinates(self) -> torch.Tensor:
        """Generate normalized 2D coordinates for an image."""
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, self.image_size),
                torch.linspace(-1, 1, self.image_size),
                indexing="ij",
            ),
            dim=-1,
        )
        return coords.unsqueeze(0)  # Add batch dimension

    def _generate_target_image(self) -> torch.Tensor:
        """Generate a target image (simple pattern for demonstration)."""
        x = torch.linspace(-1, 1, self.image_size)
        y = torch.linspace(-1, 1, self.image_size)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        R = torch.sin(2 * np.pi * X) * torch.cos(2 * np.pi * Y)
        G = torch.cos(4 * np.pi * X * Y)
        B = torch.sin(4 * np.pi * torch.sqrt(X**2 + Y**2))

        image = torch.stack([R, G, B], dim=-1)
        return image.unsqueeze(0)  # Add batch dimension

    def _load_target_image(self, image_path: str) -> torch.Tensor:
        """Load and process a target image from path."""
        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),  # Converts to [0,1] and CHW format
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Convert to [-1,1]
            ]
        )

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.permute(0, 2, 3, 1)  # Convert to NHWC format

    def __len__(self) -> int:
        return 1  # Single image dataset

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords, self.target


class LitSIREN(pl.LightningModule):
    def __init__(
        self,
        depth: int,
        hidden_dim: int,
        use_sine: bool = True,
        use_pe: bool = True,
        omega: float = 30.0,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Network parameters
        self.depth = depth
        self.omega = omega
        self.use_sine = use_sine
        self.use_pe = use_pe
        self.learning_rate = learning_rate

        # Input is always 2D coordinates
        self.in_dim = 64 if use_pe else 2
        # Output is RGB values
        self.out_dim = 3

        # Build network
        self.linears = nn.ModuleList()
        last_dim = self.in_dim

        for i in range(depth):
            cur_dim = hidden_dim if i < depth - 1 else self.out_dim
            self.linears.append(nn.Linear(last_dim, cur_dim))
            last_dim = cur_dim

        self.init_weights()

    def get_2d_sincos_pos_embed(
        self, x: torch.Tensor, embed_dim: int = 64
    ) -> torch.Tensor:
        """Convert 2D coordinates to positional embeddings."""
        # Normalize x to [0, 1] if not already
        if x.min() < 0:
            x = (x + 1) / 2
        B = x.shape[0]
        orig_shape = x.shape[:-1]
        x = x.view(B, -1, 2)  # (B, N, 2)

        dim_each = embed_dim // 2
        pe = []

        for i in range(2):  # For x and y coordinates
            pos = x[..., i].unsqueeze(-1)
            div_term = torch.exp(
                torch.arange(0, dim_each // 2, dtype=pos.dtype, device=pos.device)
                * -(math.log(10000.0) / (dim_each // 2))
            )
            pe_sin = torch.sin(pos * div_term)
            pe_cos = torch.cos(pos * div_term)
            pe_coord = torch.cat([pe_sin, pe_cos], dim=-1)
            pe.append(pe_coord)

        pe = torch.cat(pe, dim=-1)
        return pe.view(*orig_shape, embed_dim)

    def init_weights(self):
        """Initialize weights using SIREN strategy."""
        for i, layer in enumerate(self.linears):
            if i == 0:
                # First layer
                bound = 1 / layer.weight.shape[1]
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.zeros_(layer.bias)
            else:
                # Hidden layers
                if self.use_sine:
                    bound = np.sqrt(6 / layer.weight.shape[1]) / self.omega
                    nn.init.uniform_(layer.weight, -bound, bound)
                else:
                    nn.init.kaiming_uniform_(layer.weight, a=0.1)  # For LeakyReLU
                nn.init.zeros_(layer.bias)

    def activation(self, x: torch.Tensor, is_first: bool = False) -> torch.Tensor:
        """Apply either sine or leaky relu activation."""
        if self.use_sine:
            omega = self.omega
            return torch.sin(omega * x)
        else:
            return nn.functional.leaky_relu(x, negative_slope=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, query_shape = x.shape[0], x.shape[1:-1]
        x = x.view(B, -1, x.shape[-1])

        if self.use_pe:
            x = self.get_2d_sincos_pos_embed(x)

        for i in range(self.depth):
            x = self.linears[i](x)
            if i < self.depth - 1:
                x = self.activation(x, is_first=(i == 0))

        return x.view(B, *query_shape, -1)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        coords, target = batch
        output = self(coords)
        loss = nn.L1Loss()(output, target)

        # Log training metrics
        self.log("train_loss", loss, prog_bar=True)

        # Log images periodically
        if self.global_step % 100 == 0:
            # Reshape output and target for visualization (B, H, W, C) -> (B, C, H, W)
            output_img = output.squeeze().permute(2, 0, 1).unsqueeze(0)
            target_img = target.squeeze().permute(2, 0, 1).unsqueeze(0)

            # Scale from [-1, 1] to [0, 1] for visualization
            self.logger.experiment.add_images(
                "Generated Image", (output_img + 1) / 2, self.global_step
            )

            # Log target image only once
            if self.global_step == 0:
                self.logger.experiment.add_images(
                    "Target Image", (target_img + 1) / 2, 0
                )

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9)
        )
        return optimizer


def main():
    # Configuration
    IMAGE_SIZE = 256
    HIDDEN_DIM = 256
    DEPTH = 5
    TARGET_IMAGE_PATH = "/mnt/Files/workspace/HyperNerf/LDMI/data/imagenet100/train/n01440764/n01440764_438.JPEG"
    MAX_STEPS = int(1e5)

    # Create logs directory if it doesn't exist
    os.makedirs("lightning_logs", exist_ok=True)

    # Create dataset
    dataset = SIRENDataset(IMAGE_SIZE, target_image_path=TARGET_IMAGE_PATH)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Create and train models with different configurations
    configs = [
        {"use_sine": True, "use_pe": False, "name": "SIREN_Sine_NoPE"},
        {"use_sine": False, "use_pe": True, "name": "SIREN_LeakyReLU_PE"},
    ]

    for config in configs:
        print(f"\nTraining {config['name']}...")

        # Initialize model
        model = LitSIREN(
            depth=DEPTH,
            hidden_dim=HIDDEN_DIM,
            use_sine=config["use_sine"],
            use_pe=config["use_pe"],
        )

        # Initialize logger
        logger = TensorBoardLogger("lightning_logs", name=config["name"])

        # Initialize trainer
        trainer = pl.Trainer(
            max_steps=MAX_STEPS,
            accelerator="auto",
            devices=1,
            logger=logger,
            callbacks=[
                ModelCheckpoint(
                    dirpath=f"lightning_logs/{config['name']}/checkpoints",
                    filename="{epoch}-{train_loss:.4f}",
                    monitor="train_loss",
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
        print(
            f"Training completed for {config['name']}. Check TensorBoard for visualizations."
        )


if __name__ == "__main__":
    main()
    print("\nTraining completed! To view results, run:")
    print("tensorboard --logdir=lightning_logs")
    print("Then open http://localhost:6006 in your browser")
