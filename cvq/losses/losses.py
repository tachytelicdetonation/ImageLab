"""
CVQ training losses.

Faithful to the VQGAN / taming-transformers objective that VILA-U & CVQ inherit:
    L = L_recon (L1/L2) + w_perc * LPIPS + w_sem * L_semantic
        + lambda_adaptive * disc_factor * lambda_gan(c_keep) * L_GAN
plus the channel-wise VQ commitment loss (computed inside the quantizer).

Two paper-specific pieces:
  * lambda_gan(c_keep): the adaptive GAN weight from arXiv:2605.26089 eq.,
        lambda_gan(c_keep) = lambda0 / (1 + exp(-eta * (c_keep - C/2)))
    so the adversarial term is weak when few channels are kept (a coarse, blurry
    reconstruction is *expected* to look unreal) and strong when many channels are
    kept (it should be sharp). eta=0.05, lambda0=1 per the paper.
  * L_semantic: a SigLIP feature-space consistency loss honoring sem_weight=1. We
    push the frozen-SigLIP embedding of the reconstruction toward that of the input,
    preserving semantic content through the channel-wise bottleneck.

`calculate_adaptive_weight` is taming's trick: it rescales the GAN gradient to match
the perceptual-loss gradient magnitude at the decoder's last layer, which stabilizes
the recon-vs-adversarial balance without hand-tuning.
"""

from __future__ import annotations

import math

import lpips
import torch
from torch import nn
from torch.nn import functional as F


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real, logits_fake):
    return 0.5 * (
        torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake))
    )


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    """Disable a term until `threshold` steps (used to delay the GAN turn-on)."""
    return value if global_step < threshold else weight


class CVQLoss(nn.Module):
    def __init__(
        self,
        disc_start: int,                 # global step at which the GAN turns on
        recon_loss_type: str = "l1",     # "l1" or "l2" (paper states pixel-wise l2)
        perceptual_weight: float = 1.0,
        semantic_weight: float = 1.0,    # sem_weight=1 in the README
        codebook_weight: float = 1.0,    # multiplies the quantizer commitment loss
        disc_weight: float = 0.8,        # base GAN weight (lambda0 territory)
        disc_loss: str = "hinge",
        lpips_net: str = "vgg",          # "vgg" (faithful) | "alex"/"squeeze" (small)
        gan_eta: float = 0.05,           # sigmoid sharpness for lambda_gan(c_keep)
        use_adaptive_disc_weight: bool = True,
    ):
        super().__init__()
        self.recon_loss_type = recon_loss_type
        self.perceptual = lpips.LPIPS(net=lpips_net).eval()
        for p in self.perceptual.parameters():
            p.requires_grad_(False)
        self.perceptual_weight = perceptual_weight
        self.semantic_weight = semantic_weight
        self.codebook_weight = codebook_weight
        self.disc_weight = disc_weight
        self.disc_start = disc_start
        self.gan_eta = gan_eta
        self.use_adaptive_disc_weight = use_adaptive_disc_weight
        self.d_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

    # ------------------------------------------------------------------ #
    def _recon(self, recon, target):
        if self.recon_loss_type == "l2":
            return F.mse_loss(recon, target, reduction="mean")
        return F.l1_loss(recon, target, reduction="mean")

    def lambda_gan(self, c_keep: int | None, total_channels: int) -> float:
        """Paper's channel-count-aware adversarial weight (sigmoid in c_keep)."""
        if c_keep is None:
            return 1.0
        return 1.0 / (1.0 + math.exp(-self.gan_eta * (c_keep - total_channels / 2)))

    def calculate_adaptive_weight(self, nll_grad_src, g_grad_src, last_layer):
        """taming's adaptive weight: ||grad(recon)|| / ||grad(gan)|| at last layer."""
        nll_grads = torch.autograd.grad(nll_grad_src, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_grad_src, last_layer, retain_graph=True)[0]
        w = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        return torch.clamp(w, 0.0, 1e4).detach()

    # ------------------------------------------------------------------ #
    def generator_step(
        self,
        target,
        recon,
        vq_loss,
        discriminator,
        last_layer,
        global_step,
        c_keep=None,
        total_channels=256,
        siglip_real=None,
        siglip_recon_fn=None,
    ):
        """Compute the generator/autoencoder loss and a dict of components."""
        rec = self._recon(recon, target)
        p_loss = self.perceptual(recon, target).mean() if self.perceptual_weight > 0 else recon.new_zeros(())
        nll = rec + self.perceptual_weight * p_loss

        # Semantic consistency in frozen-SigLIP feature space (sem_weight).
        sem_loss = recon.new_zeros(())
        if self.semantic_weight > 0 and siglip_real is not None and siglip_recon_fn is not None:
            siglip_recon = siglip_recon_fn(recon)               # (B, C, g, g)
            r = siglip_real.flatten(1)
            s = siglip_recon.flatten(1)
            sem_loss = (1.0 - F.cosine_similarity(r, s, dim=1)).mean()

        # Adversarial term (only after disc_start).
        disc_factor = adopt_weight(1.0, global_step, threshold=self.disc_start)
        if disc_factor > 0:
            logits_fake = discriminator(recon)
            g_loss = -torch.mean(logits_fake)
            if self.use_adaptive_disc_weight and last_layer is not None:
                d_w = self.calculate_adaptive_weight(nll, g_loss, last_layer)
            else:
                d_w = torch.tensor(1.0, device=recon.device)
            lam = self.lambda_gan(c_keep, total_channels)
            gan_term = d_w * self.disc_weight * disc_factor * lam * g_loss
        else:
            g_loss = recon.new_zeros(())
            d_w = recon.new_zeros(())
            gan_term = recon.new_zeros(())

        total = (
            nll
            + self.semantic_weight * sem_loss
            + self.codebook_weight * vq_loss
            + gan_term
        )
        logs = {
            "loss/total": total.item(),
            "loss/recon": rec.item(),
            "loss/lpips": float(p_loss.item()) if torch.is_tensor(p_loss) else 0.0,
            "loss/semantic": float(sem_loss.item()) if torch.is_tensor(sem_loss) else 0.0,
            "loss/vq": vq_loss.item(),
            "loss/g_adv": float(g_loss.item()) if torch.is_tensor(g_loss) else 0.0,
            "loss/adaptive_w": float(d_w.item()) if torch.is_tensor(d_w) else 0.0,
        }
        return total, logs

    def discriminator_step(self, target, recon, discriminator, global_step):
        disc_factor = adopt_weight(1.0, global_step, threshold=self.disc_start)
        if disc_factor == 0:
            z = target.new_zeros(())
            return z, {"loss/disc": 0.0, "loss/logits_real": 0.0, "loss/logits_fake": 0.0}
        logits_real = discriminator(target.detach())
        logits_fake = discriminator(recon.detach())
        d_loss = disc_factor * self.d_loss(logits_real, logits_fake)
        logs = {
            "loss/disc": d_loss.item(),
            "loss/logits_real": logits_real.mean().item(),
            "loss/logits_fake": logits_fake.mean().item(),
        }
        return d_loss, logs
