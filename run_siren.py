import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import math
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms


class ModifiedSIREN(nn.Module):
    def __init__(
        self,
        depth: int,
        hidden_dim: int,
        use_sine: bool = True,
        use_pe: bool = True,
        omega: float = 30.0,
    ):
        super().__init__()
        self.depth = depth
        self.omega = omega
        self.use_sine = use_sine
        self.use_pe = use_pe

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
            # omega = self.omega if is_first else 1.0
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
            else:
                # x = torch.tanh(x)  # Final activation to bound output
                x = x

        return x.view(B, *query_shape, -1)


def generate_coordinates(size: int, device: torch.device) -> torch.Tensor:
    """Generate normalized 2D coordinates for an image."""
    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
        ),
        dim=-1,
    ).to(device)

    return coords.unsqueeze(0)  # Add batch dimension


def generate_target_image(size: int, device: torch.device) -> torch.Tensor:
    """Generate a target image (simple pattern for demonstration)."""
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Create a simple pattern (you can modify this)
    R = torch.sin(2 * np.pi * X) * torch.cos(2 * np.pi * Y)
    G = torch.cos(4 * np.pi * X * Y)
    B = torch.sin(4 * np.pi * torch.sqrt(X**2 + Y**2))

    image = torch.stack([R, G, B], dim=-1).to(device)
    return image.unsqueeze(0)  # Add batch dimension


def load_target_image(image_path: str, size: int, device: torch.device) -> torch.Tensor:
    """Load and process a target image from path."""
    # Load and resize image
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),  # Converts to [0,1] and CHW format
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Convert to [-1,1]
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image_tensor.permute(0, 2, 3, 1)  # Convert to NHWC format


def train_siren(
    model: ModifiedSIREN,
    image_size: int = 256,
    num_iters: int = int(1e5),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    config_name: str = "default",
    target_image_path: Optional[str] = None,
) -> Tuple[list, torch.Tensor]:
    """Train SIREN to reproduce a target image."""
    # Create tensorboard writer
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", f"{config_name}_{current_time}")
    writer = SummaryWriter(log_dir)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))
    coords = generate_coordinates(image_size, device)

    # Load target image or generate pattern
    if target_image_path and os.path.exists(target_image_path):
        target = load_target_image(target_image_path, image_size, device)
        print(f"Loaded target image from: {target_image_path}")
    else:
        target = generate_target_image(image_size, device)
        print("Using generated pattern as target")

    # Log model graph
    writer.add_graph(model, coords)

    # Log target image - permute from NHWC to NCHW
    writer.add_images("Target Image", (target.permute(0, 3, 1, 2) + 1) / 2, 0)

    losses = []
    final_output = None
    log_interval = num_iters // 100  # Log 100 times during training

    for iter in range(num_iters):
        optimizer.zero_grad()
        output = model(coords)
        loss = nn.L1Loss()(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        final_output = output.detach()

        # Log to tensorboard
        writer.add_scalar("Loss/train", loss.item(), iter)

        if (iter + 1) % log_interval == 0:
            print(f"Iteration {iter+1}/{num_iters}, Loss: {loss.item():.6f}")
            # Log images periodically - permute from NHWC to NCHW
            writer.add_images(
                "Generated Image", (output.detach().permute(0, 3, 1, 2) + 1) / 2, iter
            )

            # Log histograms of model parameters
            for name, param in model.named_parameters():
                writer.add_histogram(f"Parameters/{name}", param.data, iter)
                if param.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", param.grad.data, iter)

    writer.close()
    return losses, final_output


def visualize_results(losses: list, output: torch.Tensor, target: torch.Tensor):
    """Visualize training results."""
    plt.figure(figsize=(15, 5))

    # Plot loss curve
    plt.subplot(131)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    # Plot output image
    plt.subplot(132)
    plt.imshow(output[0].cpu().numpy())
    plt.title("SIREN Output")
    plt.axis("off")

    # Plot target image
    plt.subplot(133)
    plt.imshow(target[0].cpu().numpy())
    plt.title("Target Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # Configuration
    IMAGE_SIZE = 256
    HIDDEN_DIM = 256
    DEPTH = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_IMAGE_PATH = "/mnt/Files/workspace/HyperNerf/LDMI/data/imagenet100/train/n01440764/n01440764_438.JPEG"  # Replace with your image path

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create and train models with different configurations
    configs = [
        {"use_sine": True, "use_pe": False, "name": "SIREN_Sine_NoPE"},
        {"use_sine": False, "use_pe": True, "name": "SIREN_LeakyReLU_PE"},
    ]

    for config in configs:
        print(f"\nTraining {config['name']}...")
        model = ModifiedSIREN(
            depth=DEPTH,
            hidden_dim=HIDDEN_DIM,
            use_sine=config["use_sine"],
            use_pe=config["use_pe"],
        )

        losses, output = train_siren(
            model,
            image_size=IMAGE_SIZE,
            device=DEVICE,
            config_name=config["name"],
            target_image_path=TARGET_IMAGE_PATH,
        )

        print(
            f"Training completed for {config['name']}. Check TensorBoard for visualizations."
        )


if __name__ == "__main__":
    main()
    print("\nTraining completed! To view results, run:")
    print("tensorboard --logdir=logs")
    print("Then open http://localhost:6006 in your browser")
