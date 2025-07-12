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

class VQModel(pl.LightningModule):
    def __init__(self,
                 encoder,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 decoder=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = instantiate_from_config(encoder)
        # If not specified, we use symmetric decoder
        if decoder is None:
            self.decoder = Decoder(**encoder["params"])
        else:
            self.decoder = instantiate_from_config(decoder)
        self.loss = instantiate_from_config(lossconfig, extra_args={'n_classes': n_embed})
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(encoder["params"]["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, encoder["params"]["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x, quantize=False):
        h = self.encoder(x)
        h = self.quant_conv(h)
        if quantize:
            quant, emb_loss, info = self.quantize(h)
            return quant, emb_loss, info
        return h


    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant, *args, **kwargs):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, *args, **kwargs)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input, quantize=True)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        try:
            x = batch[k]
            if len(x.shape) == 3:
                x = x[..., None]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        except:
            x = batch[0].to(memory_format=torch.contiguous_format).float()
        # Old from ldm repo:
        """x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()"""
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
    
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae, sync_dist=True)
        self.log_dict(log_dict_disc, sync_dist=True)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class VAE(pl.LightningModule):
    def __init__(self,
                 encoder,
                 lossconfig,
                 embed_dim,
                 decoder=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 #mask_input=False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = instantiate_from_config(encoder)
        # If not specified, we use symmetric decoder
        if decoder is None:
            self.decoder = Decoder(**encoder["params"])
        else:
            self.decoder = instantiate_from_config(decoder)
        
        #self.mask_input = mask_input
            
        self.loss = instantiate_from_config(lossconfig)
        self.quant_conv = torch.nn.Conv2d(2*encoder["params"]["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, encoder["params"]["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, *args, **kwargs):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, *args, **kwargs)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        try:
            x = batch[k]
            if len(x.shape) == 3:
                x = x[..., None]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        except:
            x = batch[0]
        # Old from ldm repo:
        """x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()"""
        return x

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        data = self.get_input(batch, self.image_key)

        if hasattr(self, 'data_converter'):
            inputs = self.apply_data_converter(data)
            if hasattr(self.data_converter, "batch_to_coordinates_and_features"):
                data = inputs
        else:
            inputs = data
        
        """if self.mask_input:
            bs, f, *dims = inputs.shape
            mask = (torch.rand(bs,1,*dims, device=inputs.device)) > torch.rand(1, device=inputs.device) * 0.99
        """
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(data, reconstructions, posterior, 
                                            optimizer_idx=optimizer_idx, 
                                            global_step=self.global_step,
                                            last_layer=self.get_last_layer(), 
                                            split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(data, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        data = self.get_input(batch, self.image_key)

        if hasattr(self, 'data_converter'):
            inputs = self.apply_data_converter(data)
            if hasattr(self.data_converter, "batch_to_coordinates_and_features"):
                data = inputs
        else:
            inputs = data

        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(data, reconstructions, posterior, 
                                        optimizer_idx=0, 
                                        global_step=self.global_step,
                                        last_layer=self.get_last_layer(), 
                                        split="val")
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], sync_dist=True)
        self.log_dict(log_dict_ae, sync_dist=True)        
        
        if hasattr(self.loss, 'discriminator'):
            discloss, log_dict_disc = self.loss(data, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")
            self.log_dict(log_dict_disc, sync_dist=True)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        
        if hasattr(self.loss, 'discriminator'):
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))

            return [opt_ae, opt_disc], []
        else:
            return opt_ae

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)

        if not only_inputs:
            xrec, posterior = self(x, sample_posterior=False)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x



class IPVAE(pl.LightningModule):
    """ A VAE for INR modelling point clouds"""
    def __init__(self,
                 encoder,
                 decoder,
                 lossconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 monitor=None,
                 ):
        
        super().__init__()
        self.encoder = instantiate_from_config(encoder)
        self.decoder = instantiate_from_config(decoder)

        self.quant_conv = torch.nn.Conv2d(2*self.encoder.dim_z, 2*self.encoder.dim_z, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.encoder.dim_z, self.encoder.dim_z, 1)

        self.loss = instantiate_from_config(lossconfig)
        
        self.pretrained = ckpt_path is not None
        self.quantize = None
        self.force_not_quantize = True

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def forward(self, input, sample_posterior=True):

        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.decode(z)

        return dec, posterior

    def encode(self, x):
        #split x into coordinates and features

        h = self.encoder(x).contiguous()

        # Output of the encoder os (bs, latent_dim)
        # Convert latent to image of shape (latent_shape)
        # This is cheating. The latent_dim will be actually 1, 
        # and we will have a "row" image with size latent dim.
        # However, the Transformer will handle that.
        moments = self.quant_conv(h).contiguous()

        posterior = DiagonalGaussianDistribution(moments)
        # h = self.quant_conv(h)
        return posterior

    def decode(self, h, coord=None):

        # also go through quantization layer
        quant = self.post_quant_conv(h).contiguous()
        
        dec = self.decoder(quant, coord=coord)

        return dec
    
    def training_step(self, batch, batch_idx):

        inputs = self.get_input(batch)
        logits, posterior = self(inputs)
        logits = logits.reshape(*inputs.shape)

        total_loss, log_dict_ae = self.loss(inputs, logits, posterior, weights=None, split = 'train')
        self.log("train/total_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        return total_loss


    def validation_step(self, batch, batch_idx):

        inputs = self.get_input(batch)

        logits, posterior = self(inputs)
        logits = logits.reshape(*inputs.shape)
        total_loss, log_dict_ae = self.loss(inputs, logits, posterior, weights=None, split = 'val')
        
        self.log("val/total_loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
    
    def get_input(self, batch):
        x = batch[0]
        return x


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))

        return [opt_ae], []


    
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, super_reconstruction=True, **kwargs):
        log = dict()
        # coordinates, features = batch
        inputs = self.get_input(batch)
        if not only_inputs:
            xrec, posterior = self(inputs, sample_posterior=False)
            xrec = torch.sigmoid(xrec)
            xrec = xrec.reshape(*inputs.shape).cpu()

            samples = torch.sigmoid(self.decode(torch.randn_like(posterior.sample())))
            samples = samples.reshape(*inputs.shape).cpu()

            log["samples"] = samples
            log["reconstructions"] = xrec

            if super_reconstruction:
                resolution = [64,64,64]
                coords = make_coord_grid(resolution, (-1, 1)).to(self.device)
                xrec_super = torch.sigmoid(self.decode(posterior.mode(), coords)).reshape([xrec.shape[0], 1, *resolution])
                log["super_reconstructions"] = xrec_super

            #denorm both 
            #log["samples"] = self.apply_data_deconverter(log["samples"]).cpu()
            #log["reconstructions"] = self.apply_data_deconverter(log["reconstructions"]).cpu()
            
        log["inputs"] = inputs.cpu()
        return log
    



class IVAE(VAE):
    def __init__(self,
                 encoder,
                 lossconfig,
                 embed_dim,
                 decoder=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 data_converter=None,
                 #mask_input=False,
                 ):
        super().__init__(
                    encoder=encoder, 
                    lossconfig=lossconfig, 
                    embed_dim=embed_dim, 
                    ignore_keys=ignore_keys,
                    image_key=image_key,
                    colorize_nlabels=colorize_nlabels,
                    monitor=monitor,
                    #mask_input=mask_input
                    )

        
        # Change decoder to INR generator
        self.decoder = instantiate_from_config(decoder)

        if data_converter != None:
            self.data_converter = instantiate_from_config(data_converter)
            if hasattr(self.data_converter, 'coordinates'):
                coords = self.data_converter.coordinates
                self.decoder.register_buffer('shared_coord', coords, persistent=False)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        print(f"Encoder Parameters: {sum(p.numel() for p in self.encoder.parameters()) / 1e6:.2f}M")
        print(f"Decoder Parameters: {sum(p.numel() for p in self.decoder.parameters()) / 1e6:.2f}M")


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def apply_data_converter(self, x):
        if hasattr(self, "data_converter"):
            if hasattr(self.data_converter, "batch_to_coordinates_and_features"):
                _, features = self.data_converter.batch_to_coordinates_and_features(x)
            else:
                features = self.data_converter(x)
        return features
    
        
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, super_resolution=True, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)

        log["inputs"] = x

        if hasattr(self, 'data_converter'):
            _, x = self.data_converter.batch_to_coordinates_and_features(x)

        if not only_inputs:
            posterior = self.encode(x)
            z = posterior.mode()

            # Original size
            xrec = self.decode(z)

            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)

            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec

            if super_resolution:
                # Super reconstruction
                res = 2 * self.encoder.resolution
                coord = make_coord_grid((res,res), (-1, 1)).to(x.device)
                xrec_super = self.decode(z, coord=coord)
                if x.shape[1] > 3:
                    xrec_super = self.to_rgb(xrec_super)
                log["super_reconstructions"] = xrec_super
                
        return log



class IVQModel(VQModel):
    def __init__(self,
                 encoder,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 decoder=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 data_converter=None,
                 ):
        super().__init__(encoder=encoder, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels, monitor=monitor)
        
        # Change decoder to INR generator
        self.decoder = instantiate_from_config(decoder) 

        if data_converter != None:
            self.data_converter = instantiate_from_config(data_converter)
            if hasattr(self.data_converter, 'coordinates'):
                coords = self.data_converter.coordinates
                self.decoder.register_buffer('shared_coord', coords, persistent=False)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        print(f"Encoder Parameters: {sum(p.numel() for p in self.encoder.parameters()) / 1e6:.2f}M")
        print(f"Decoder Parameters: {sum(p.numel() for p in self.decoder.parameters()) / 1e6:.2f}M")
        print(f"Loss Parameters: {sum(p.numel() for p in self.loss.parameters()) / 1e6:.2f}M")


    def apply_data_converter(self, x):
        if hasattr(self, "data_converter"):
            if hasattr(self.data_converter, "batch_to_coordinates_and_features"):
                _, features = self.data_converter.batch_to_coordinates_and_features(x)
            else:
                features = self.data_converter(x)
        return features
    
        
    def get_last_layer(self):
        return self.decoder.get_last_layer()


class IVQModelInterface(IVQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
