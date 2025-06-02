

import os
import torch
from utils.viz import plot_voxels_batch_new
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader


class ImageLoggerExperiment:
    def __init__(self, max_images=16, cols=4, rescale=True, shuffle=True):
        self.max_images = max_images
        self.cols = cols
        self.rescale = rescale
        self.shuffle = shuffle

    def run(self, model, datamodule, logdir):
        """
        Runs image logging after training
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print("Running Image Logger Experiment...")

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
        
        # Get images
        with torch.no_grad():
            images = model.log_images(batch, only_inputs=False, N=self.max_images)

        # Directory for logging
        img_dir = os.path.join(logdir, 'images', 'test')
        os.makedirs(img_dir, exist_ok=True)

        for k in images:
            images[k] = torch.clamp(images[k], -1, 1)
            if k == 'super_reconstructions':
                padding = 4
            else:
                padding = 2
            grid = torchvision.utils.make_grid(images[k], nrow=self.cols, padding=padding, pad_value=1)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.cpu().numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = f"{k}.png"
            path = os.path.join(img_dir, filename)
            Image.fromarray(grid).save(path)
