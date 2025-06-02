

import os
import torch
from utils.viz.render import *
from torchvision.utils import save_image

class VoxelLoggerExperiment:
    def __init__(self, max_images=16, cols=4, threshold=0.5, point_cloud=False, cubify=True, smooth=True):
        self.max_images = max_images
        self.cols = cols
        self.threshold = threshold
        self.point_cloud = point_cloud
        self.cubify = cubify
        self.smooth = smooth

    def run(self, model, datamodule, logdir):
        """
        Runs voxel logging after training, similar to VoxelLogger's log_voxel function.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print("Running Voxel Logger Experiment...")

        dataloader = datamodule.val_dataloader()

        # Get validation batch
        batch = next(iter(dataloader))  # Get first batch
        bs = batch[0].shape[0]
        while bs != self.max_images:
            batch_ = next(iter(dataloader))  # Get first batch
            batch[0] = torch.cat([batch[0], batch_[0]])[:self.max_images]
            batch[1] = torch.cat([batch[1], batch_[1]])[:self.max_images]
            bs = batch[0].shape[0]
        
        batch = [b.to(model.device) for b in batch]
                
        with torch.no_grad():
            images = model.log_images(batch, only_inputs=False, super_resolution=False, 
                                      plot_diffusion_rows=False,
                                      plot_progressive_rows=False,
                                      plot_denoise_rows=False
                                      )

        # Directory for logging
        img_dir = os.path.join(logdir, 'images', 'test')
        os.makedirs(img_dir, exist_ok=True)

        # Save voxel visualizations
        for key in images: 
            filename = os.path.join(img_dir, f"{key}.png")
            
            # Convert voxels to mesh and render
            if self.cubify:
                mesh = voxels_to_cubified_mesh(images[key])
            else:
                mesh = voxels_to_torch3d_mesh(images[key], self.smooth)
                    

            mesh = mesh.to(model.device)
            print("Completed voxel to mesh conversion.")
            images_out = render_mesh(model.device, mesh, flat=self.cubify)
            print("Completed rendering.")

            # Save
            save_image(images_out, filename, nrow=self.cols, pad_value=1.)

        print("Voxel logging experiment completed.")
        