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
        head_type: str = "softmax",
        mbm_depth: int = 2,
        mbm_heads: int = 8,
        mbm_infer_steps: int = 4,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.num_channels = num_channels
        self.codebook_size = codebook_size
        self.backbone_dtype = backbone_dtype
        self.head_type = head_type

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
        nn.init.normal_(self.image_embed.weight, std=0.02)
        nn.init.normal_(self.boi, std=0.02)
        self.image_embed.to(backbone_dtype)
        self.boi.data = self.boi.data.to(backbone_dtype)
        # Output head: flat K-way softmax (EOSTok/Fork A) or masked-bit modeling (BAR/Fork B).
        if head_type == "mbm":
            from .mbm_head import MBMHead
            self.mbm = MBMHead(self.hidden, codebook_size, depth=mbm_depth,
                               n_heads=mbm_heads, n_infer_steps=mbm_infer_steps)
        else:
            self.image_head = nn.Linear(self.hidden, codebook_size, bias=False)
            nn.init.normal_(self.image_head.weight, std=0.02)
            self.image_head.to(backbone_dtype)

    # ------------------------------------------------------------------ #
    def _run_backbone(self, inputs_embeds, attention_mask, past_key_values=None, use_cache=False):
        """Return last hidden states (B, T, hidden) for a causal pass.

        When use_cache=True, also returns past_key_values for incremental decoding.
        Used by the cached generate() path: text+BOI is run once to seed the cache, then
        each subsequent step feeds a single new embedding -> O(C) total backbone work
        instead of O(C^2) (vs the old re-run-the-whole-sequence loop).
        """
        out = self.decoder_lm(
            inputs_embeds=inputs_embeds.to(self.backbone_dtype),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if use_cache:
            past = getattr(out, "past_key_values", None)
            return hidden, past
        return hidden

    def _logits(self, hidden):
        """Image head in fp32 for stable cross-entropy / softmax."""
        return self.image_head(hidden).float()

    def _img_hidden(self, text_ids, text_mask, image_idxs):
        """Teacher-forced backbone pass -> (B, C, H) hidden states at image positions.

        Sequence [text, BOI, img_1..img_{C-1}] so position c predicts channel c. Head-agnostic:
        both the softmax and MBM heads consume these hidden states."""
        B, L = text_ids.shape
        C = image_idxs.shape[1]
        te = self.text_embed(text_ids)                            # (B, L, H) backbone dtype
        boi = self.boi.expand(B, 1, self.hidden)
        ie = self.image_embed(image_idxs)                         # (B, C, H)
        seq = torch.cat([te.to(self.backbone_dtype), boi, ie[:, :-1]], dim=1)  # (B, L+C, H)
        img_mask = torch.ones(B, C, device=text_ids.device, dtype=text_mask.dtype)
        attn = torch.cat([text_mask, img_mask], dim=1)            # (B, L+C)
        hidden = self._run_backbone(seq, attn)                    # (B, L+C, H)
        return hidden[:, L:, :]                                   # (B, C, H)

    def forward(self, text_ids, text_mask, image_idxs, return_hidden=False):
        """Softmax-head teacher-forced logits (B, C, K). MBM uses `ar_loss` instead."""
        img_hidden = self._img_hidden(text_ids, text_mask, image_idxs)
        logits = self._logits(img_hidden)                        # (B, C, K) fp32
        if return_hidden:
            return logits, img_hidden
        return logits

    def ar_loss(self, text_ids, text_mask, image_idxs, channel_weights=None):
        """Unified AR objective -> (loss, logs, aux).

        softmax head: (optionally channel-weighted) NTP cross-entropy; aux={'logits': (B,C,K)}
                      so train_e2e can run the EOSTok APR soft-decode.
        mbm head:     masked-bit BCE over each channel-token's bits (BAR); aux={} (the soft
                      codebook APR is not defined for bit-prediction — disable lambda_apr)."""
        B, C = image_idxs.shape
        img_hidden = self._img_hidden(text_ids, text_mask, image_idxs)    # (B, C, H)
        if self.head_type == "mbm":
            ctx = img_hidden.reshape(B * C, self.hidden).float()
            tgt = image_idxs.reshape(B * C)
            loss, hlogs = self.mbm(ctx, tgt)
            with torch.no_grad():
                tok_acc = self.mbm.exact_match(ctx, tgt)
            logs = {"car/ntp_loss": loss.item(), "car/bit_acc": hlogs["bit_acc"],
                    "car/token_acc": tok_acc.item()}
            return loss, logs, {}
        logits = self._logits(img_hidden)                                # (B, C, K) fp32
        K = logits.shape[-1]
        ce_pt = F.cross_entropy(logits.reshape(-1, K), image_idxs.reshape(-1),
                                reduction="none").reshape(B, C)
        if channel_weights is None:
            loss = ce_pt.mean()
        else:
            loss = (ce_pt * channel_weights.to(ce_pt.device)[None, :]).mean()
        with torch.no_grad():
            acc = (logits.argmax(-1) == image_idxs).float().mean()
            cprefix = max(1, C // 4)
            acc_prefix = (logits[:, :cprefix].argmax(-1) ==
                          image_idxs[:, :cprefix]).float().mean()
        logs = {"car/ntp_loss": loss.item(), "car/token_acc": acc.item(),
                "car/token_acc_prefix": acc_prefix.item()}
        return loss, logs, {"logits": logits}

    def loss(self, text_ids, text_mask, image_idxs, channel_weights: torch.Tensor | None = None):
        """NTP cross-entropy over image channels (EOSTok's L_NTP).

        If `channel_weights` is given (shape (C,), mean=1), the per-channel CE is reweighted
        so early (coarse) channels are penalized more -- the formal coupling between CVQ's
        coarse-to-fine channel ordering and EOSTok's flat NTP loss. mean(w)=1 keeps the
        overall scale equal to the unweighted loss.
        """
        logits = self(text_ids, text_mask, image_idxs)            # (B, C, K) fp32
        B, C, K = logits.shape
        ce_pt = F.cross_entropy(
            logits.reshape(-1, K), image_idxs.reshape(-1), reduction="none",
        ).reshape(B, C)
        if channel_weights is None:
            loss = ce_pt.mean()
        else:
            loss = (ce_pt * channel_weights.to(ce_pt.device)[None, :]).mean()
        with torch.no_grad():
            acc = (logits.argmax(-1) == image_idxs).float().mean()
        return loss, {"car/ntp_loss": loss.item(), "car/token_acc": acc.item()}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, text_ids, text_mask, temperature=1.0, top_k=0,
                 cfg_scale=1.0, uncond_text_ids=None, uncond_text_mask=None,
                 use_cache: bool = True):
        """Autoregressively sample C channel-tokens conditioned on text.

        Classifier-free guidance: if cfg_scale>1, logit = uncond + cfg_scale*(cond - uncond).
        Returns: (B, C) sampled indices, ready for tokenizer.lookup -> decode.

        Default uses HF's KV cache (O(C) total backbone work). Pass use_cache=False to
        fall back to the old no-cache path -- kept primarily as a sanity check that the
        cached path produces the same logits.
        """
        if use_cache:
            return self._generate_cached(
                text_ids, text_mask, temperature, top_k, cfg_scale,
                uncond_text_ids, uncond_text_mask,
            )
        return self._generate_nocache(
            text_ids, text_mask, temperature, top_k, cfg_scale,
            uncond_text_ids, uncond_text_mask,
        )

    def _sample_one(self, logits, temperature, top_k):
        logits = logits / max(temperature, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(logits, top_k, dim=-1)
            logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
        return torch.multinomial(logits.softmax(-1), 1)

    def _sample_step(self, last_h, u_last_h, temperature, top_k, cfg_scale, do_cfg):
        """Sample one channel-token index (B,1) from the current hidden state(s).

        softmax: K-way (optionally CFG-mixed, top-k) sampling. mbm: iterative masked-bit
        refinement via the MBM head (CFG mixes bit logits inside the head)."""
        if self.head_type == "mbm":
            if do_cfg:
                idx = self.mbm.generate_cfg(last_h.float(), u_last_h.float(),
                                            cfg_scale, temperature)
            else:
                idx = self.mbm.generate(last_h.float(), temperature)
            return idx.unsqueeze(1)
        logits = self._logits(last_h)
        if do_cfg:
            ulogits = self._logits(u_last_h)
            logits = ulogits + cfg_scale * (logits - ulogits)
        return self._sample_one(logits, temperature, top_k)

    @torch.no_grad()
    def _generate_cached(self, text_ids, text_mask, temperature, top_k, cfg_scale,
                         uncond_text_ids, uncond_text_mask):
        """KV-cache path: text+BOI is processed once, then one embed/step thereafter."""
        B = text_ids.shape[0]
        dev = text_ids.device
        do_cfg = cfg_scale != 1.0 and uncond_text_ids is not None

        def seed(ids, mask):
            te = self.text_embed(ids)
            boi = self.boi.expand(B, 1, self.hidden)
            seq = torch.cat([te.to(self.backbone_dtype), boi], dim=1)
            am = torch.cat([mask, torch.ones(B, 1, device=dev, dtype=mask.dtype)], 1)
            hidden, past = self._run_backbone(seq, am, past_key_values=None, use_cache=True)
            return hidden[:, -1, :], past, am

        last_h, past, attn = seed(text_ids, text_mask)
        upast = uattn = None
        if do_cfg:
            u_last_h, upast, uattn = seed(uncond_text_ids, uncond_text_mask)

        out_idxs = []
        for c in range(self.num_channels):
            nxt = self._sample_step(last_h, u_last_h if do_cfg else None,
                                    temperature, top_k, cfg_scale, do_cfg)   # (B, 1)
            out_idxs.append(nxt)
            if c == self.num_channels - 1:
                break
            emb = self.image_embed(nxt)                               # (B, 1, H)
            attn = torch.cat([attn, torch.ones(B, 1, device=dev, dtype=attn.dtype)], 1)
            hidden, past = self._run_backbone(emb, attn, past_key_values=past, use_cache=True)
            last_h = hidden[:, -1, :]
            if do_cfg:
                uattn = torch.cat([uattn, torch.ones(B, 1, device=dev, dtype=uattn.dtype)], 1)
                u_hidden, upast = self._run_backbone(emb, uattn, past_key_values=upast, use_cache=True)
                u_last_h = u_hidden[:, -1, :]
        return torch.cat(out_idxs, dim=1)                             # (B, C)

    @torch.no_grad()
    def _generate_nocache(self, text_ids, text_mask, temperature, top_k, cfg_scale,
                          uncond_text_ids, uncond_text_mask):
        """Original O(C^2) path. Kept as a sanity reference for the cached path."""
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
            last_h = self._run_backbone(seq, attn)[:, -1, :]
            u_last_h = self._run_backbone(useq, uattn)[:, -1, :] if do_cfg else None
            nxt = self._sample_step(last_h, u_last_h, temperature, top_k, cfg_scale, do_cfg)
            out_idxs.append(nxt)
            emb = self.image_embed(nxt)
            seq = torch.cat([seq, emb], dim=1)
            attn = torch.cat([attn, torch.ones(B, 1, device=dev, dtype=attn.dtype)], 1)
            if do_cfg:
                useq = torch.cat([useq, emb], dim=1)
                uattn = torch.cat([uattn, torch.ones(B, 1, device=dev, dtype=uattn.dtype)], 1)
        return torch.cat(out_idxs, dim=1)

    def soft_embed(self, p_hat):
        """Embed a SOFT distribution over codes: (B,C,K) -> (B,C,H). Used by the APR path so
        gradient flows from generation back through the image embedding."""
        return p_hat.to(self.image_embed.weight.dtype) @ self.image_embed.weight

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
