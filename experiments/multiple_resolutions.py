

import os
import torch
from utils.viz import plot_voxels_batch_new
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from utils.geometry import make_coord_grid
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

class MultipleResolutions:
    def __init__(self, max_images=4, resolutions=[0.25, 0.5, 1, 2, 4], patch_size=32, orig_cx=128, orig_cy=180, rescale=True, shuffle=True):
        self.max_images = max_images
        self.resolutions = resolutions
        self.patch_size = patch_size
        self.orig_cx = orig_cx
        self.orig_cy = orig_cy
        self.rescale = rescale
        self.shuffle = shuffle

    def run(self, model, datamodule, logdir):
        """
        Runs image logging after training
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print("Running Multiple Resolutions Experiment...")

        dataloader = datamodule.val_dataloader()

        # Extract its arguments and modify 'shuffle'
        dataloader = DataLoader(
            dataset=dataloader.dataset,  # Reuse dataset
            batch_size=dataloader.batch_size,
            shuffle=self.shuffle,  # Modify this
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            collate_fn=dataloader.collate_fn,
            worker_init_fn=dataloader.worker_init_fn,
            persistent_workers=dataloader.persistent_workers if hasattr(dataloader, 'persistent_workers') else False,
        )
        # Get validation batch
        batch = next(iter(dataloader))  # Get first batch
        if isinstance(batch, dict):
            bs = batch['image'].shape[0]
        else:
            bs = batch[0].shape[0]
        while bs != self.max_images:
            batch_ = next(iter(dataloader))  # Get first batch
            if isinstance(batch, dict):
                for k in batch:
                    if type(batch[k]) == torch.Tensor:
                        b = torch.cat([batch[k], batch_[k]])[:self.max_images]
                    else:
                        b = batch[k] + batch_[k]
                    b = b[:self.max_images]
                    batch[k] = b
                bs = batch['image'].shape[0]
            else:
                for i, (b, b_) in enumerate(zip(batch, batch_)):
                    b = torch.cat([b, b_])[:self.max_images]
                    batch[i] = b
                bs = b.shape[0]
        
        # Move to device
        if isinstance(batch, dict):
            for k in batch:
                if type(batch[k]) == torch.Tensor:
                    batch[k] = batch[k].to(model.device)   
        else:
            batch = [b.to(model.device) for b in batch if type(b) == torch.Tensor]
        
        size = batch[0].shape[-1]

        # Get images
        with torch.no_grad():

            # Posterior samples
            print("Sampling from the posterior...")
            z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=self.max_images)
            # Prior samples
            print("Sampling from the prior...")
            # get denoise row
            with model.ema_scope("Plotting"):
                z_samples, _ = model.sample_log(cond=c,batch_size=self.max_images,ddim=True,
                                                        ddim_steps=200,eta=1.)
            samples = model.decode_first_stage(z_samples)

            images = {
                'rec_res_1': xrec,
                'sample_res_1': samples,
                'orig': x
            }
            for res in self.resolutions:
                # Reconstruction
                print(f'Generating super-reconstructions at resolution {res}...')
                if res != 1: 
                    # Super reconstruction
                    super_size = int(res * size)
                    coord = make_coord_grid((super_size,super_size), (-1, 1)).to(x.device)

                    xrec_super = model.decode_first_stage(z, coord=coord)
                    images[f'rec_res_{res}'] = xrec_super

                    x_samples_super = model.decode_first_stage(z_samples, coord=coord)
                    images[f'sample_res_{res}'] = x_samples_super

            
        # Directory for logging
        img_dir = os.path.join(logdir, 'images', 'test', 'multiple_resolutions')
        os.makedirs(img_dir, exist_ok=True)

        for k in images:
            images[k] = torch.clamp(images[k], -1, 1)

            for b,im in enumerate(images[k]):
                grid = im
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.cpu().numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = f"b{b}_{k}.png"
                path = os.path.join(img_dir, filename)

                # Create a matplotlib figure
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(grid)
                ax.axis('off')                
                # If super-resolution, add a zoom-in box
                orig_patch_size = self.patch_size
                orig_cx, orig_cy = self.orig_cx, self.orig_cy  # center of image (can be changed)
                if "res_" in k and float(k.split("res_")[-1]) >= 1:
                    res_factor = float(k.split("res_")[-1])


                    # Get image size
                    h, w, _ = grid.shape

                    res_factor = w / size

                    # Define patch size as a fraction of output image
                    patch_size = int(w * (orig_patch_size / size))

                    # Scale original center position into current resolution
                    cx = int(orig_cx * res_factor)
                    cy = int(orig_cy * res_factor)

                    # Define crop region
                    x1 = cx - patch_size // 2
                    x2 = cx + patch_size // 2
                    y1 = cy - patch_size // 2
                    y2 = cy + patch_size // 2

                    # Compute zoom so that patch fills 1/4 of image (assuming square image and square inset)
                    inset_target_fraction = 0.5  # fraction of full image width/height to use for inset
                    zoom_factor = (w * inset_target_fraction) / patch_size  # dynamically computed zoom

                    # Create zoomed inset
                    axins = zoomed_inset_axes(ax, zoom=zoom_factor, loc=4, borderpad=0)  # loc=3 is bottom-left
                    axins.imshow(grid)
                    axins.set_xlim(x1, x2)
                    axins.set_ylim(y2, y1)  # flip y-axis
                    axins.axis('off')

                    import matplotlib.patches as patches
                    # Add black border rectangle in normalized inset axes coordinates (from (0,0) to (1,1))
                    rect = patches.Rectangle(
                        (0, 0), 1, 1,
                        transform=axins.transAxes,
                        linewidth=1,
                        edgecolor='black',
                        facecolor='none'
                    )
                    axins.add_patch(rect)

                # Save image
                plt.savefig(path, bbox_inches='tight', pad_inches=0)
                plt.close()
                plt.cla()
        