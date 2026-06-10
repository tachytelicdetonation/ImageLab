"""Importing cvq.models registers every built-in component (see cvq/registry.py).

Add your experimental module to this import list (or import it yourself before building)
so its @register decorators run.
"""

from . import decoder        # noqa: F401  registers decoder/vqgan
from . import encoder_cnn    # noqa: F401  registers encoder/cnn
from . import fsq            # noqa: F401  registers quantizer/fsq
from . import heads          # noqa: F401  registers ar_head/{softmax,mbm}
from . import lfq            # noqa: F401  registers quantizer/lfq (the worked example)
from . import quantizer      # noqa: F401  registers quantizer/ibq

from .car import CAR                        # noqa: F401
from .discriminator import NLayerDiscriminator  # noqa: F401
from .tokenizer import CVQTokenizer         # noqa: F401
