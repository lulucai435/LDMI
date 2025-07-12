import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


class GaussianLoss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0):

        super().__init__()
        self.kl_weight = kl_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)


    def forward(self, inputs, reconstructions, posteriors, split="train", **kwargs):
        
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        mse = ((inputs - reconstructions) ** 2).view(inputs.shape[0], -1).mean(dim=1)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            
        loss = weighted_nll_loss + self.kl_weight * kl_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/mse".format(split): mse.detach().mean(),
                "{}/psnr".format(split): psnr.detach().mean(),
                }
        return loss, log


