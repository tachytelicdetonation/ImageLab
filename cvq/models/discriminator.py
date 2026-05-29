"""
PatchGAN discriminator (NLayerDiscriminator), the standard pix2pix / taming-transformers
adversarial critic used by VQGAN and inherited by VILA-U / CVQ.

It classifies overlapping ~70x70 patches as real/fake rather than the whole image,
which is what gives VQGAN its crisp local texture. Output is a spatial map of logits.
"""

from __future__ import annotations

import functools

import torch
from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3, use_actnorm: bool = False):
        super().__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        use_bias = False  # BatchNorm has affine bias

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)
