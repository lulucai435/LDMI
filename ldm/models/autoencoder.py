import os
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.decoders import *
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config

from utils.geometry import make_coord_grid
from packaging import version
import time
from pdb import set_trace as bb
import torchvision

class VQModel(pl.LightningModule):

    def __init__(
        self,
        encoder,
        decoder,
        image_key="image",
        monitor=None,
        **kwargs,
    ):
        super().__init__()
        self.image_key = image_key
        # self.encoder = instantiate_from_config(encoder)
        self.decoder = instantiate_from_config(decoder)

        if monitor is not None:
            self.monitor = monitor

        # Store learning rate from kwargs
        self.learning_rate = kwargs.get("learning_rate", 1e-4)

    def decode(self, h):
        # h is ignored since we're testing decoder.inr directly
        dec = self.decoder(data=None)  # TransINR will use shared_coord
        return dec

    def forward(self, input):
        dec = self.decode(None)
        return dec

    def get_input(self, batch, k):
        try:
            x = batch[k]
            if len(x.shape) == 3:
                x = x[..., None]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        except:
            x = batch[0].to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec = self(x)

        # Simple L1 loss
        loss = torch.abs(x - xrec).mean()

        # Log images every 100 steps
        if self.global_step % 100 == 0:
            torchvision.utils.save_image(
                (torch.cat([x, xrec], dim=0) + 1) / 2,
                os.path.join(self.logger.log_dir, f"train_{self.global_step:06d}.png"),
            )

        self.log(
            "train/loss", loss, prog_bar=True, logger=True, on_step=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec = self(x)

        # Simple L1 loss
        loss = torch.abs(x - xrec).mean()

        self.log(
            "val/loss", loss, prog_bar=True, logger=True, on_step=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        # Use Adam with better defaults for INR training
        optimizer = torch.optim.Adam(
            self.decoder.inr.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.9),  # Same as successful SIREN implementation
            eps=1e-8,
        )
        return optimizer

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log
