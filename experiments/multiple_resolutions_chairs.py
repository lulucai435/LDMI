

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
from utils.viz.render import *
from torchvision.utils import save_image


class MultipleResolutionsChairs:
    def __init__(self, max_images=4, resolutions=[0.25, 0.5, 1, 2, 4], rescale=True, shuffle=True):
        self.max_images = max_images
        self.resolutions = resolutions
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
                print(f'Generating super-reconstructions at resolution {res}...')
                if res != 1: 
                                
                    super_size = int(res * size)
                    super_res = [super_size, super_size, super_size]
                    coord = model.first_stage_model.data_converter.superresolve_coordinates(super_res).to(model.device)
                    # Repeat for batch_size
                    coord = coord.unsqueeze(0).repeat([xrec.shape[0], 1, 1])
                    xrec_super = torch.sigmoid(model.first_stage_model.decode(z, coord)).reshape([xrec.shape[0], 1, *super_res])
                    images[f'rec_res_{res}'] = xrec_super

                    x_samples_super = torch.sigmoid(model.decode_first_stage(z_samples, coord=coord)).reshape([xrec.shape[0], 1, *super_res])
                    images[f'sample_res_{res}'] = x_samples_super

            
        # Directory for logging
        img_dir = os.path.join(logdir, 'images', 'test', 'multiple_resolutions')
        os.makedirs(img_dir, exist_ok=True)

        for k in images:
            for b,im in enumerate(images[k]):
                filename = f"b{b}_{k}.png"
                path = os.path.join(img_dir, filename)

                try:
                    # Convert voxels to mesh and render
                    mesh = voxels_to_cubified_mesh(im.unsqueeze(0))

                    mesh = mesh.to(model.device)
                    print("Completed voxel to mesh conversion.")
                    images_out = render_mesh(model.device, mesh, flat=True)
                    print("Completed rendering.")

                    # Save
                    save_image(images_out, path, nrow=1)
                except:
                    print("Problematic image")
            