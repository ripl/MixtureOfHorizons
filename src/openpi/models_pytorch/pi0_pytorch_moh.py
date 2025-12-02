import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from typing import Callable, Optional, Union
from transformers.cache_utils import DynamicCache


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
        time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim

        # Dense layer for adaptive normalization (if cond_dim is provided)
        if cond_dim is not None:
            # self.dense = nn.Linear(cond_dim, dim * 3, bias=True, dtype=torch.bfloat16)
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            # Initialize with zeros (matches source implementation)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
            self.dense = None

    def _norm(self, x):
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(self, x, cond=None):
        dtype = x.dtype  # original dtype, could be half-precision
        normed_inputs = self._norm(x)

        if cond is None or self.dense is None:
            # regular RMSNorm
            # scale by learned parameter in float32 (matches source implementation)
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None  # return in original dtype with None gate

        # adaptive RMSNorm (if cond is provided and dense layer exists)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")

        # self.dense.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        modulation = self.dense(cond)
        # Reshape modulation to broadcast properly: [batch, 1, features] for [batch, seq, features]
        if len(x.shape) == 3:  # [batch, seq, features]
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)

        return normed_inputs.to(dtype), gate.to(dtype)


class PI0Pytorch(nn.Module):
    def __init__(self, config, horizons, use_gate_noise=True):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.horizons = horizons
        self.use_gate_noise = use_gate_noise
        print(f"horizons: {horizons}")

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.gate_out_proj = nn.Linear(action_expert_config.width, 1)
        if self.use_gate_noise:
            self.gate_noise_layer = nn.Linear(action_expert_config.width, 1)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")
        self.softplus = nn.Softplus()
        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
        self.mean_fusion = False

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
            self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep, action_pad_mask=None):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        if action_pad_mask is None:
            action_pad_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_pad_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (noisy_actions.shape[1] - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, observation, actions, loss_config: dict = None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        batch_size = actions.shape[0]

        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(batch_size, actions.device)
        time = time.unsqueeze(0).expand(len(self.horizons), -1)

        x_t = time[:, :, None, None] * noise.unsqueeze(0) + (1 - time[:, :, None, None]) * actions.unsqueeze(0)
        u_t = noise - actions

        # STAGE 1: VLM Prefix Pass (Compute KV cache once)
        # =================================================
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        if self.paligemma_with_expert.paligemma.language_model.layers[
            0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Perform a forward pass only on the VLM part to get the KV cache
        (prefix_output, _), prefix_past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],  # Pass only prefix embeddings
            use_cache=True,
        )

        # STAGE 2: Action Head Suffix Passes (Parallelized via batching)
        # ==============================================================
        num_horizons, max_horizon = len(self.horizons), self.horizons[-1]

        batched_prefix_pad_masks = prefix_pad_masks.repeat_interleave(num_horizons, dim=0)
        batched_prefix_att_masks = prefix_att_masks.repeat_interleave(num_horizons, dim=0)

        batched_past_key_values = self._repeat_past_key_values(prefix_past_key_values, num_horizons)
        batched_past_key_values = DynamicCache.from_legacy_cache(past_key_values=batched_past_key_values)

        batched_state = state.repeat_interleave(num_horizons, dim=0)

        # Reshape x_t and time to align with the new batch dimension.
        # x_t: (num_h, bsize, max_h, dim) -> (bsize, num_h, max_h, dim) -> (bsize * num_h, max_h, dim)
        batched_x_t = x_t.permute(1, 0, 2, 3).reshape(batch_size * num_horizons, max_horizon, -1)
        # time: (num_h, bsize) -> (bsize, num_h) -> (bsize * num_h)
        batched_time = time.permute(1, 0).reshape(batch_size * num_horizons)

        # Create a padding mask for actions, as each head has a different valid horizon length.
        # Shape: (num_h, max_h)
        action_pad_mask = torch.arange(max_horizon, device=actions.device)[None, :] < \
                          torch.tensor(self.horizons, device=actions.device)[:, None]
        # Shape: (num_h, bsize, max_h)
        action_pad_mask = action_pad_mask.unsqueeze(1).expand(-1, batch_size, -1)
        # Shape: (bsize * num_h, max_h)
        batched_action_pad_mask = action_pad_mask.permute(1, 0, 2).reshape(batch_size * num_horizons, max_horizon)

        # Embed the batched suffix inputs.
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            batched_state, batched_x_t, batched_time, action_pad_mask=batched_action_pad_mask
        )

        if prefix_embs.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Combine prefix and suffix masks for cross-attention for the entire batch.
        pad_masks = torch.cat([batched_prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([batched_prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_embs.shape[1]

        # Create position IDs for the suffix part only.
        suffix_position_ids = torch.arange(prefix_len, prefix_len + suffix_len, device=actions.device).unsqueeze(0)
        suffix_att_2d_masks_4d = full_att_2d_masks_4d[:, :, -suffix_len:, :]

        # Perform a single forward pass for the suffix, reusing the prefix's KV cache.
        # The model is expected to handle the batched input and apply the correct expert internally.
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=suffix_att_2d_masks_4d,
            position_ids=suffix_position_ids,
            past_key_values=batched_past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # Project the output and extract the action predictions.
        v_t_batched = self.action_out_proj(suffix_out.to(torch.float32))
        action_start_index = 0 if self.pi05 else 1
        v_t_actions_padded = v_t_batched[:, action_start_index: action_start_index + max_horizon, :]

        # Reshape the output to separate predictions for each head.
        # -> (bsize, num_h, max_h, dim) -> (num_h, bsize, max_h, dim)
        all_v_t_preds = v_t_actions_padded.view(
            batch_size, num_horizons, max_horizon, -1
        ).permute(1, 0, 2, 3)

        # Calculate losses in a parallelized manner.
        all_head_losses = []
        for i, h in enumerate(self.horizons):
            v_t_head = all_v_t_preds[i, :, :h, :]
            target_v_t = u_t[:, :h, :]
            head_loss = F.mse_loss(v_t_head, target_v_t)
            all_head_losses.append(head_loss)

        # 1. Primary Loss: Ensures each expert head is trained well.
        individual_loss = torch.sum(torch.stack(all_head_losses))

        # 2. Mixture Loss: Trains the gating network to combine predictions effectively.
        if self.mean_fusion:
            gate_logits = torch.ones(
                (batch_size, max_horizon, num_horizons),
                device=suffix_out.device,
                dtype=suffix_out.dtype
            )
        else:
            gate_logits = self.gate_out_proj(suffix_out.to(torch.float32))
            gate_logits = gate_logits[:, action_start_index: action_start_index + max_horizon,
                          :]  # B * num_horizons, max_horizon, 1

            if self.use_gate_noise:
                # Add Learnable Noise to gate logits
                noise_epsilon = 1e-2
                raw_noise_stddev = self.gate_noise_layer(suffix_out.to(torch.float32))
                raw_noise_stddev = raw_noise_stddev[:, action_start_index: action_start_index + max_horizon, :]
                noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
                gate_logits = gate_logits + (torch.randn_like(gate_logits) * noise_stddev)

                gate_logits = gate_logits.reshape(batch_size, num_horizons, max_horizon).permute(0, 2, 1)

        valid_heads_mask = torch.tensor(
            [[step < h for h in self.horizons] for step in range(max_horizon)],
            device=actions.device,
            dtype=torch.bool
        ).unsqueeze(0)

        masked_gate_logits = torch.where(valid_heads_mask, gate_logits, torch.finfo(gate_logits.dtype).min)
        gate_weights = F.softmax(masked_gate_logits, dim=-1)

        # Combine predictions using the dynamic weights.
        # all_v_t_preds: (num_h, bsize, max_h, dim) -> (bsize, num_h, max_h, dim)
        all_v_t_preds_padded = all_v_t_preds.permute(1, 0, 2, 3)
        # gate_weights: (1, max_h, num_h) -> (bsize, num_h, max_h, 1)
        # combined_preds: (bsize, max_h, dim)
        v_t_combined = (gate_weights.permute(0, 2, 1).unsqueeze(-1) * all_v_t_preds_padded).sum(dim=1)

        aux_loss_weight = loss_config.get("aux_weight", 1) if loss_config else 1
        auxiliary_loss = F.mse_loss(v_t_combined, u_t)

        # 3. Balance Loss: Encourage the gate layer to output weights flexibly
        loss_components = []
        boundaries = sorted(list(set([0] + self.horizons)))
        for i in range(len(boundaries) - 1):
            start_step, end_step = boundaries[i], boundaries[i + 1]
            active_expert_indices = [idx for idx, h in enumerate(self.horizons) if h > start_step]

            if len(active_expert_indices) > 1:
                segment_gate_weights = gate_weights[:, start_step:end_step, :]
                active_expert_weights = segment_gate_weights[:, :, active_expert_indices]
                avg_expert_prob_in_segment = active_expert_weights.mean(dim=(0, 1))
                segment_loss = self.cv_squared(avg_expert_prob_in_segment)
                loss_components.append(segment_loss)

        load_balancing_loss = torch.mean(torch.stack(loss_components))
        balance_loss_weight = loss_config.get("balance_weight", 0.001) if loss_config else 0.001

        total_loss = individual_loss + aux_loss_weight * auxiliary_loss + balance_loss_weight * load_balancing_loss

        return total_loss

    def _repeat_past_key_values(self, past_key_values, n_repeats):
        """Helper method to repeat past_key_values for batched inference."""
        if past_key_values is None:
            return None
        repeated_pkv = []
        for layer_pkv in past_key_values:
            k, v = layer_pkv
            # k, v shapes: [bsize, num_heads, seq_len, head_dim]
            repeated_k = k.repeat_interleave(n_repeats, dim=0)
            repeated_v = v.repeat_interleave(n_repeats, dim=0)
            repeated_pkv.append((repeated_k, repeated_v))
        return tuple(repeated_pkv)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, ret_weights=False,
                       use_dynamic_replanning=False, scale_ratio=1,
                       min_replan_steps=5, min_active_horizons=3):
        """
        Do a full inference forward and compute the action by a cascading weighted average.
        This version is optimized to avoid redundant embedding and padding steps.
        """
        bsize = observation.state.shape[0]
        max_horizon = self.horizons[-1]
        num_horizons = len(self.horizons)

        if noise is None:
            actions_shape = (bsize, max_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # STAGE 1: VLM Prefix Pass (Compute KV cache once)
        # =====================================================================
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens,
                                                                            lang_masks)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        if self.paligemma_with_expert.paligemma.language_model.layers[
            0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # STAGE 2: Iterative Denoising with Batched Horizons
        # =============================================================
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        gate_weights_to_log = []

        batched_prefix_pad_masks = prefix_pad_masks.repeat_interleave(num_horizons, dim=0)
        batched_prefix_att_masks = prefix_att_masks.repeat_interleave(num_horizons, dim=0)
        batched_past_key_values = self._repeat_past_key_values(past_key_values, num_horizons)
        batched_past_key_values = DynamicCache.from_legacy_cache(past_key_values=batched_past_key_values)
        batched_state = state.repeat_interleave(num_horizons, dim=0)

        if use_dynamic_replanning:
            # shape: [bsize, num_horizons, max_horizon]
            l1_disagreement_sum = torch.zeros(
                (bsize, max_horizon),
                dtype=torch.float32,
                device=device,
            )

        while time >= -dt / 2:
            expanded_time = time.expand(bsize * num_horizons)
            padded_x_t_list, action_pad_mask_list = [], []
            for h in self.horizons:
                padded_x_t = F.pad(x_t[:, :h, :], (0, 0, 0, max_horizon - h))
                padded_x_t_list.append(padded_x_t)
                pad_mask = F.pad(torch.ones((bsize, h), device=device, dtype=torch.bool), (0, max_horizon - h),
                                 value=False)
                action_pad_mask_list.append(pad_mask)

            batched_x_t = torch.cat(padded_x_t_list, dim=0)
            action_pad_mask = torch.cat(action_pad_mask_list, dim=0)

            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                batched_state, batched_x_t, expanded_time, action_pad_mask=action_pad_mask)

            if prefix_embs.dtype == torch.bfloat16:
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

            pad_masks = torch.cat([batched_prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([batched_prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            full_att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            prefix_len = prefix_pad_masks.shape[1]
            suffix_len = suffix_embs.shape[1]
            suffix_position_ids = torch.arange(prefix_len, prefix_len + suffix_len, device=device).unsqueeze(0)
            suffix_att_2d_masks_4d = full_att_2d_masks_4d[:, :, -suffix_len:, :]

            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=suffix_att_2d_masks_4d,
                position_ids=suffix_position_ids,
                past_key_values=batched_past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            action_start_index = 0 if self.pi05 else 1

            if self.mean_fusion:
                gate_logits = torch.ones(
                    (bsize, max_horizon, num_horizons),
                    device=suffix_out.device,
                    dtype=suffix_out.dtype
                )
            else:
                gate_logits = self.gate_out_proj(suffix_out.to(torch.float32))
                gate_logits = gate_logits[:, action_start_index: action_start_index + max_horizon, :]  # B * num_horizons, max_horizon, 1
                gate_logits = gate_logits.reshape(bsize, num_horizons, max_horizon).permute(0, 2, 1)

            valid_heads_mask = torch.tensor(
                [[step < h for h in self.horizons] for step in range(max_horizon)],
                device=device, dtype=torch.bool
            ).unsqueeze(0)

            masked_gate_logits = torch.where(valid_heads_mask, gate_logits, torch.finfo(gate_logits.dtype).min)
            gate_weights = F.softmax(masked_gate_logits, dim=-1)

            if ret_weights:
                gate_weights_to_log.append(torch.round(gate_weights, decimals=3))

            v_t_batched = self.action_out_proj(suffix_out.to(torch.float32))
            v_t_actions_padded = v_t_batched[:, action_start_index: action_start_index + max_horizon]

            # [B * H, L, D] -> [B, H, L, D]
            all_v_t_preds_padded = v_t_actions_padded.view(num_horizons, bsize, max_horizon, -1).permute(1, 0, 2, 3)

            # Combine predictions using the dynamic weights. The mask in gate_weights handles everything.
            v_t = (gate_weights.permute(0, 2, 1).unsqueeze(-1) * all_v_t_preds_padded).sum(dim=1)
            
            if use_dynamic_replanning:
                # x_prev = x_t  # [B, L, D]
                fused_iter = x_t + dt * v_t  # [B, L, D]
                indiv_iter = x_t.unsqueeze(1) + dt * all_v_t_preds_padded  # [B, H, L, D]

                # L1 over action dim -> [B, H, L]
                per_iter_l1 = torch.sum(
                    torch.abs(indiv_iter - fused_iter.unsqueeze(1)),
                    dim=-1,
                )
                per_iter_l1 = (per_iter_l1 * gate_weights.permute(0, 2, 1)).sum(dim=1)
                l1_disagreement_sum += per_iter_l1

            # Euler update
            time += dt
            x_t = x_t + dt * v_t

        fused_actions = x_t

        return_dict = {}

        if use_dynamic_replanning:
            assert bsize == 1, "Dynamic replanning currently supports batch size 1."

            dynamic_replan_steps = min_replan_steps
            dynamic_l1_threshold = l1_disagreement_sum[0, :min_replan_steps].mean() * scale_ratio

            # determine the executable steps based on step-wise mean disagreement
            for step in range(min_replan_steps, max_horizon):
                active_indices = [h_idx for h_idx, h in enumerate(self.horizons) if step < h]
                valid_horizons_for_this_step = len(active_indices)
                if valid_horizons_for_this_step <= min_active_horizons:
                    break

                if l1_disagreement_sum[0, step] < dynamic_l1_threshold:
                    dynamic_replan_steps += 1
                else:
                    break

            final_replan_steps = max(min_replan_steps, dynamic_replan_steps)
            return_dict["actions"] = fused_actions[:, :final_replan_steps, :]
            return_dict["replan_steps"] = final_replan_steps
        else:
            return_dict["actions"] = fused_actions

        if ret_weights and len(gate_weights_to_log) > 0:
            return_dict["gate_weights"] = torch.stack(gate_weights_to_log, dim=1).detach().cpu()

        return return_dict
