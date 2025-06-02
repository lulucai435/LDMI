

import os
import torch
import numpy as np
from utils.viz.render import *
from torchvision.utils import save_image


class RenderChairs:
    def __init__(self, voxels_path, max_images=6, cols=6,
                smooth=True, cubify=True):
        self.voxels_path = voxels_path
        self.max_images = max_images
        self.cols = cols
        self.smooth = smooth
        self.cubify = cubify

    def run(self, model, datamodule, logdir):
        """
        Runs image logging after training
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print("Rendering Chairs...")

        data = np.load(self.voxels_path, allow_pickle=True)

        filename = os.path.basename(self.voxels_path)

        voxels = {
            f"reconstructions_{filename}":    torch.Tensor(data[0]).to(model.device)[:self.max_images],
            f"missing_{filename}":   torch.Tensor(data[1]).to(model.device)[:self.max_images],
            f"input_{filename}":    torch.Tensor(data[2]).to(model.device)[:self.max_images]
        }

        # Directory for logging
        img_dir = os.path.join(logdir, 'images', 'test')
        os.makedirs(img_dir, exist_ok=True)

        # Reconstructions
        for k in voxels:
            # Convert voxels to mesh and render
            if self.cubify:
                mesh = voxels_to_cubified_mesh(voxels[k])
            else:
                mesh = voxels_to_torch3d_mesh(voxels[k], self.smooth)
                
            mesh = mesh.to(model.device)
            print("Completed voxel to mesh conversion.")
            images = render_mesh(model.device, mesh, flat=self.cubify)
            print("Completed rendering.")

            # Save
            path = os.path.join(img_dir, f"{k}.png")
            save_image(images, path, nrow=self.cols, pad_value=1.)

