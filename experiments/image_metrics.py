

import os
import torch
from utils.viz import plot_voxels_batch_new
import torchvision
import numpy as np
from tqdm import tqdm
from experiments.inception import InceptionV3
from experiments.fid_score import *
import pandas as pd

class ImageMetrics:
    def __init__(self, save_name):
        self.save_name = save_name

    def run(self, model, datamodule, logdir):
        """
        Runs image logging after training
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print("Running Image Logger Experiment...")

        try: 
            dataloader = datamodule.test_dataloader()
        except:
            print('No test set available. Using validation set')
            dataloader = datamodule.val_dataloader()

        metrics = {
            'psnr': [],
            'fid': [],
        }

        # Load inception
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        inception_model = InceptionV3([block_idx]).to(device)

        save_path = f'/zhome/d8/1/207127/LDMI_pre/experiments/results/{self.save_name}_metrics.csv'

        # Initialize tqdm with dynamic metric tracking
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing Metrics")
        
        for b, batch in progress_bar:
            # Move to device
            if isinstance(batch, dict):
                for k in batch:
                    if type(batch[k]) == torch.Tensor:
                        batch[k] = batch[k].to(model.device)   
            else:
                batch = [b.to(model.device) for b in batch if type(b) == torch.Tensor]
                
            with torch.no_grad():
                images = model.log_images(batch, 
                                        quantize_denoise=False, 
                                        inpaint=False, 
                                        plot_denoise_rows=False,
                                        plot_progressive_rows=False,
                                        plot_diffusion_rows=False,
                                        split='test')
                samples = images['samples']
                samples = torch.clamp(samples, -1, 1)
                reconstructions = images['reconstructions']
                reconstructions = torch.clamp(reconstructions, -1, 1)
                inputs = images['inputs']
                inputs = torch.clamp(inputs, -1, 1)

            bs = inputs.shape[0]

            inputs_norm = (inputs + 1) / 2
            reconstructions_norm = (reconstructions + 1) / 2
            samples_norm = (samples + 1) / 2

            mses = ((inputs_norm - reconstructions_norm) ** 2).view(inputs_norm.shape[0], -1).mean(dim=-1)
            psnr = (10 * torch.log10(1.0 / mses)).cpu().numpy()
            
            act1_ = get_activations(inputs_norm, inception_model)
            act2_ = get_activations(samples_norm, inception_model)

            if b==0:
                metrics['psnr'] = psnr
                act1 = act1_
                act2 = act2_
            else:
                metrics['psnr'] = np.concatenate((metrics['psnr'], psnr))
                act1 = np.concatenate((act1, act1_), axis=0)
                act2 = np.concatenate((act2, act2_), axis=0)

            m1 = np.mean(act1, axis=0)
            s1 = np.cov(act1, rowvar=False)

            m2 = np.mean(act2, axis=0)
            s2 = np.cov(act2, rowvar=False)

            fid = calculate_frechet_distance(m1, s1, m2, s2)

            metrics['fid'] = np.full_like(metrics['psnr'], fid)

            # Convert dictionary to dataframe
            df = pd.DataFrame(metrics)

            # Append to CSV (create header only if the file doesn't exist)
            df.to_csv(save_path, mode='a', index=False, header=not os.path.exists(save_path))
            
            # Update tqdm bar with metrics
            progress_bar.set_postfix(psnr_mean=f"{metrics['psnr'].mean():.4f}", 
                                     psnr_std=f"{metrics['psnr'].std():.4f}", 
                                     fid=f"{metrics['fid'].mean():.4f}")  


        print(f"Results saved at: {save_path}")
