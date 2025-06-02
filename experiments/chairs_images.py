

import os
import torch
import numpy as np
from utils.viz.render import *
from torchvision.utils import save_image


class ChairsImages:
    def __init__(self, max_images=6, cols=6,
                smooth=True, cubify=True):
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

        print("Running Chairs Logger Experiment...")

        dataloader = datamodule.test_dataloader()
        batch = next(iter(dataloader))
        batch = [b.to(model.device)[:self.max_images] for b in batch if type(b) == torch.Tensor]
                
        # Get images
        with torch.no_grad():
            voxels = model.log_images(batch, only_inputs=False)

        print("Completed voxel generation.")

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

