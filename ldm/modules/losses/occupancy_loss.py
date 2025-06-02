import torch
import torch.nn as nn

class BernoulliLoss(nn.Module):
    def __init__(self, kl_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.kl_weight = kl_weight
        
    def forward(self, inputs, logits, posteriors, weights=None, split = 'train'):

        # inputs are binary and expressed as in {0,1}
        nll_loss = self.bce(logits, inputs)
        rec_loss = torch.abs(inputs.contiguous() - torch.sigmoid(logits).contiguous())

        if weights is not None:
            nll_loss = weights * nll_loss

        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        
        kl_loss = posteriors.kl() #because z has 2 dims, sums over the second dim
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]


        loss = nll_loss + self.kl_weight * kl_loss 

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                }
        return loss, log