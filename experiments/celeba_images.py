

import os
import torch
import torchvision
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class CelebAImages:
    def __init__(self, max_images=16, cols=4, rescale=True):
        self.max_images = max_images
        self.cols = cols
        self.rescale = rescale

    def run(self, model, datamodule, logdir):
        """
        Runs image logging after training
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print("Running Image Logger Experiment...")

        #dataloader = datamodule.val_dataloader()
        train_dataloader = datamodule.train_dataloader()
        train_dataset = train_dataloader.dataset
        val_dataloader = datamodule.val_dataloader()
        val_dataset = val_dataloader.dataset

        # Get batch from VAMoH
        # Loading in CelebA style
        filenames = ['184484.jpg', '013498.jpg', '039472.jpg', '047444.jpg', '085437.jpg', '042581.jpg']
        # Loading in CelebA-HQ style
        #filenames = ['28007.jpg', '28008.jpg', '28009.jpg', '28010.jpg', '28011.jpg', '28012.jpg']
        bs = len(filenames)

        """transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])"""

        batch = []
        for f in filenames:
            #x = Image.open(os.path.join(val_dataset.root, val_dataset.base_folder, 'img_align_celeba', f)) # Celeba    
            try:
                x = Image.open(os.path.join(train_dataset.data_root, 'female', f)) #CelebaHQ
            except:
                try:
                    x = Image.open(os.path.join(train_dataset.data_root, 'male', f)) #CelebaHQ    
                except:
                    try:
                        x = Image.open(os.path.join(val_dataset.data_root, 'female', f)) #CelebaHQ
                    except:
                        x = Image.open(os.path.join(val_dataset.data_root, 'male', f)) #CelebaHQ
                    
            #x = Image.open(os.path.join('/zhome/d8/1/207127/LDMI_pre/experiments/figs/', f))
            #x = transform(x)
            x = val_dataset.transform(x)
            batch.append(x)
        batch = torch.stack(batch)
        
        batch = (batch.to(model.device),)
        
        # Get images
        with torch.no_grad():
            images = model.log_images(batch, only_inputs=False, N=self.max_images)

        # Directory for logging
        img_dir = os.path.join(logdir, 'images', 'test')
        os.makedirs(img_dir, exist_ok=True)

        for k in images:
            #images[k] = torch.stack([2 * (img - img.min()) / (img.max() - img.min()) - 1 for img in images[k]])
            images[k] = torch.clamp(images[k], -1, 1)
            if k == 'super_reconstructions':
                padding = 4
            else:
                padding = 2
            grid = torchvision.utils.make_grid(images[k], nrow=bs, padding=padding, pad_value=-1)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.cpu().numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = f"{k}.png"
            path = os.path.join(img_dir, filename)
            Image.fromarray(grid).save(path)
