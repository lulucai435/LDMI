

import os
import torch
import numpy as np


class ERA5Images:
    def __init__(self, max_images=4, cols=4, rescale=True):
        self.max_images = max_images
        self.cols = cols

    def run(self, model, datamodule, logdir):
        """
        Runs image logging after training
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print("Running ERA5 Logger Experiment...")

        dataloader = datamodule.test_dataloader()
        images_save = {}
        for n, batch in enumerate(dataloader):

            batch = [b.to(model.device) for b in batch if type(b) == torch.Tensor]
                    
            # Get images
            with torch.no_grad():
                images = model.log_images(batch, only_inputs=False)

            for k in images:
                if n==0:
                    if k == 'inputs':
                        images_save[k] = images[k].cpu().numpy()
                    else:
                        im = torch.cat([images['inputs'][:,:-1], images[k]], 1)
                        images_save[k] = im.cpu().numpy()
                else:
                    if k == 'inputs':
                        images_save[k] = np.concatenate( (images_save[k], images[k].cpu().numpy()) )
                    else:
                        im = torch.cat([images['inputs'][:,:-1], images[k]], 1)
                        images_save[k] = np.concatenate( (images_save[k], im.cpu().numpy() ))

        # Directory for logging
        img_dir = os.path.join(logdir, 'images', 'test')
        os.makedirs(img_dir, exist_ok=True)

        # Save to a .npz file
        path = os.path.join(img_dir, "era5_images.npz")
        np.savez(path, **images_save)
