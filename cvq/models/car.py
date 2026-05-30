"""
CAR — Channel-wise AutoRegressive text-to-image model (EOSTok-style integration).

The CVQ tokenizer turns an image into a sequence of C=256 channel-tokens, each an index
in [0, K), ordered coarse-to-fine (the nested-dropout ordering). Generation is therefore
next-CHANNEL prediction: predict channel c+1 given the text prompt and channels 1..c.

We fuse a text LLM and the image-token AR into ONE causal sequence through a single
transformer backbone (Chameleon / LlamaGen-T2I style):

    [ name tokens ]  [BOI]  [ img_1 ... img_C ]
    └ LLM text embed ┘      └ image embed (K -> hidden) ┘
            │  one causal transformer over the whole sequence  │
       (loss ignored)        (cross-entropy: predict img_{c+1} | name, img_<=c)

Backbone-agnostic: works with a plain CausalLM (Qwen3-0.6B-Base) OR a hybrid
linear-attention / multimodal model (Qwen3.5-0.8B, model_type qwen3_5). For the latter:
  * we drive the TEXT decoder via get_decoder() (skips the vision tower),
  * hidden size is read from the embedding (multimodal configs nest it under text_config),
  * the linear-attention (Gated-DeltaNet) layers hold bf16 weights and reject fp32 inputs,
    so the backbone runs in bf16 and we cast inputs_embeds to the backbone dtype; only the
    final logits/CE are upcast to fp32 for numerical stability.

Design choices:
  * Separate image embedding + head (NOT tied to the LLM's text vocab): image tokens are a
    different modality and K != text vocab. A learned BOI embedding marks text->image.
  * Backbone trainable by default at a low LR, with `freeze_backbone` to switch to a
    frozen-LLM / adapter regime.

This is EOSTok's NTP objective. The APR loss (decode teacher-forced predictions to pixels +
unfreeze the tokenizer for true joint E2E) is implemented in cvq/train_e2e.py, using
forward(..., return_hidden=True) -> image_head over the soft prediction.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CAR(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        num_channels: int = 256,
        qwen_name: str = "Qwen/Qwen3.5-0.8B",
        freeze_backbone: bool = False,
        attn_impl: str = "sdpa",
        backbone_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.num_channels = num_channels
        self.codebook_size = codebook_size
        self.backbone_dtype = backbone_dtype

        # ---- LLM backbone (text embeddings + transformer stack) ----
        # bf16: hybrid linear-attention layers (qwen3_5) require it; plain LMs tolerate it.
        backbone = AutoModelForCausalLM.from_pretrained(
            qwen_name, dtype=backbone_dtype, trust_remote_code=True,
            attn_implementation=attn_impl,
        )
        self.backbone = backbone
        # get_decoder() returns the text-only stack even for multimodal models.
        self.decoder_lm = backbone.get_decoder()
        self.text_embed = backbone.get_input_embeddings()         # (vocab, hidden)
        self.hidden = self.text_embed.embedding_dim
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.freeze_backbone = freeze_backbone

        # ---- image modality: embedding, begin-of-image marker, output head ----
        # Match the backbone dtype so concatenated embeddings are homogeneous.
        self.image_embed = nn.Embedding(codebook_size, self.hidden)
        self.boi = nn.Parameter(torch.zeros(1, 1, self.hidden))
        self.image_head = nn.Linear(self.hidden, codebook_size, bias=False)
        nn.init.normal_(self.image_embed.weight, std=0.02)
        nn.init.normal_(self.boi, std=0.02)
        nn.init.normal_(self.image_head.weight, std=0.02)
        self.image_embed.to(backbone_dtype)
        self.image_head.to(backbone_dtype)
        self.boi.data = self.boi.data.to(backbone_dtype)

    # ------------------------------------------------------------------ #
    def _run_backbone(self, inputs_embeds, attention_mask):
        """Return last hidden states (B, T, hidden) for a full causal pass."""
        out = self.decoder_lm(
            inputs_embeds=inputs_embeds.to(self.backbone_dtype),
            attention_mask=attention_mask,
            use_cache=False,
        )
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def _logits(self, hidden):
        """Image head in fp32 for stable cross-entropy / softmax."""
        return self.image_head(hidden).float()

    def forward(self, text_ids, text_mask, image_idxs, return_hidden=False):
        """Teacher-forced next-channel prediction.

        Args:
            text_ids:   (B, L) LLM token ids of the prompt (right-padded).
            text_mask:  (B, L) 1 for real text tokens, 0 for pad.
            image_idxs: (B, C) channel-token indices from the (frozen or joint) tokenizer.
        Returns:
            logits: (B, C, K) fp32 prediction for each image position.
            (optionally) hidden at image positions (for inspection / APR).
        """
        B, L = text_ids.shape
        C = image_idxs.shape[1]
        te = self.text_embed(text_ids)                            # (B, L, H) backbone dtype
        boi = self.boi.expand(B, 1, self.hidden)
        ie = self.image_embed(image_idxs)                         # (B, C, H)
        # [text, BOI, img_1..img_{C-1}] -> predicts img_1..img_C.
        seq = torch.cat([te.to(self.backbone_dtype), boi, ie[:, :-1]], dim=1)  # (B, L+C, H)
        img_mask = torch.ones(B, C, device=text_ids.device, dtype=text_mask.dtype)
        attn = torch.cat([text_mask, img_mask], dim=1)            # (B, L+C)

        hidden = self._run_backbone(seq, attn)                    # (B, L+C, H)
        img_hidden = hidden[:, L:, :]                             # (B, C, H)
        logits = self._logits(img_hidden)                        # (B, C, K) fp32
        if return_hidden:
            return logits, img_hidden
        return logits

    def loss(self, text_ids, text_mask, image_idxs):
        """NTP cross-entropy over image channels (EOSTok's L_NTP)."""
        logits = self(text_ids, text_mask, image_idxs)            # (B, C, K) fp32
        loss = F.cross_entropy(
            logits.reshape(-1, self.codebook_size), image_idxs.reshape(-1)
        )
        with torch.no_grad():
            acc = (logits.argmax(-1) == image_idxs).float().mean()
        return loss, {"car/ntp_loss": loss.item(), "car/token_acc": acc.item()}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, text_ids, text_mask, temperature=1.0, top_k=0,
                 cfg_scale=1.0, uncond_text_ids=None, uncond_text_mask=None):
        """Autoregressively sample C channel-tokens conditioned on text.

        Classifier-free guidance: if cfg_scale>1, logit = uncond + cfg_scale*(cond - uncond).
        Returns: (B, C) sampled indices, ready for tokenizer.lookup -> decode.
        """
        B = text_ids.shape[0]
        dev = text_ids.device
        do_cfg = cfg_scale != 1.0 and uncond_text_ids is not None

        te = self.text_embed(text_ids)
        boi = self.boi.expand(B, 1, self.hidden)
        seq = torch.cat([te.to(self.backbone_dtype), boi], dim=1)
        attn = torch.cat([text_mask, torch.ones(B, 1, device=dev, dtype=text_mask.dtype)], 1)
        if do_cfg:
            ute = self.text_embed(uncond_text_ids)
            useq = torch.cat([ute.to(self.backbone_dtype), self.boi.expand(B, 1, self.hidden)], dim=1)
            uattn = torch.cat([uncond_text_mask,
                               torch.ones(B, 1, device=dev, dtype=text_mask.dtype)], 1)

        out_idxs = []
        for _ in range(self.num_channels):
            logits = self._logits(self._run_backbone(seq, attn)[:, -1, :])    # (B, K) fp32
            if do_cfg:
                ulogits = self._logits(self._run_backbone(useq, uattn)[:, -1, :])
                logits = ulogits + cfg_scale * (logits - ulogits)
            logits = logits / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(logits.softmax(-1), 1)        # (B, 1)
            out_idxs.append(nxt)
            emb = self.image_embed(nxt)                           # (B, 1, H)
            seq = torch.cat([seq, emb], dim=1)
            attn = torch.cat([attn, torch.ones(B, 1, device=dev, dtype=attn.dtype)], 1)
            if do_cfg:
                useq = torch.cat([useq, emb], dim=1)
                uattn = torch.cat([uattn, torch.ones(B, 1, device=dev, dtype=uattn.dtype)], 1)
        return torch.cat(out_idxs, dim=1)                         # (B, C)

    def soft_embed(self, p_hat):
        """Embed a SOFT distribution over codes: (B,C,K) -> (B,C,H). Used by the APR path so
        gradient flows from generation back through the image embedding."""
        return p_hat.to(self.image_embed.weight.dtype) @ self.image_embed.weight

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
