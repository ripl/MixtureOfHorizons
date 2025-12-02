"""
This file implements a regression-based version of PI0 with a gated
multi-horizon ensemble, based on pi0reg.py and
pi0_pytorch_moh.py.

Instead of diffusion, it uses a set of learnable query tokens as input
to an ensemble of action experts, each focused on a different horizon.
It performs direct regression to the target actions using L1 loss.
A learnable gating network combines the predictions from all experts.
"""

import logging
import math
from typing import Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from transformers.cache_utils import DynamicCache


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


class PI0Pytorch(nn.Module):
    def __init__(self, config, horizons):
        super().__init__()
        self.config = config
        self.horizons = horizons
        if not horizons or horizons != sorted(horizons):
            raise ValueError(f"Horizons must be a non-empty, sorted list. Got: {horizons}")
        self.max_horizon = horizons[-1]

        print(f"Initializing PI0RegressionGateHead with horizons: {horizons}")

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, False],  # No AdaRMS/time conditioning for regression
            precision=config.dtype,
        )

        # Learnable tokens to query the action expert
        # These will be expanded to [bsize * num_heads, max_horizon, width]
        self.action_query_embedding = nn.Parameter(
            torch.randn(1, 1, action_expert_config.width)
        )

        # Projection for proprioceptive state
        self.state_proj = nn.Linear(32, action_expert_config.width)

        # Final projection from expert hidden state to action dimension
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        # Gating network layers (from pi0_pytorch_moh.py)
        self.gate_out_proj = nn.Linear(action_expert_config.width, 1)
        self.gate_noise_layer = nn.Linear(action_expert_config.width, 1)
        self.softplus = nn.Softplus()

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

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

    def embed_suffix(self, state, action_pad_mask=None):
        """
        Embed state and learnable action queries to prepare for Expert Gemma processing.
        This is a regression-based suffix embedder.
        """
        embs = []
        pad_masks = []
        att_masks = []

        bsize = state.shape[0]
        device = state.device

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        # Embed state
        def state_proj_func(state):
            return self.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state)

        embs.append(state_emb[:, None, :])
        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Expand learnable action queries to the max horizon
        action_queries = self.action_query_embedding.expand(
            bsize, self.max_horizon, -1
        )
        embs.append(action_queries)

        if action_pad_mask is None:
            action_pad_mask = torch.ones(
                bsize, self.max_horizon, dtype=torch.bool, device=device
            )
        pad_masks.append(action_pad_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.max_horizon - 1))

        # No time conditioning for regression
        adarms_cond = None

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def cv_squared(self, x):
        """Coefficient of variation squared, for load balancing loss."""
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, observation, actions, loss_config: dict = None) -> Tensor:
        """
        Do a full training forward pass and compute the composite L1 loss.
        'actions' must be the ground-truth actions, padded to self.max_horizon.
        Shape: (batch_size, self.max_horizon, action_dim)
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        batch_size = actions.shape[0]
        num_horizons = len(self.horizons)

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
        batched_prefix_pad_masks = prefix_pad_masks.repeat_interleave(num_horizons, dim=0)
        batched_prefix_att_masks = prefix_att_masks.repeat_interleave(num_horizons, dim=0)

        batched_past_key_values = self._repeat_past_key_values(prefix_past_key_values, num_horizons)
        batched_past_key_values = DynamicCache.from_legacy_cache(past_key_values=batched_past_key_values)

        batched_state = state.repeat_interleave(num_horizons, dim=0)

        # Create a padding mask for actions, as each head has a different valid horizon length.
        # Shape: (num_h, max_h)
        action_pad_mask = torch.arange(self.max_horizon, device=actions.device)[None, :] < \
                          torch.tensor(self.horizons, device=actions.device)[:, None]
        # Shape: (num_h, bsize, max_h)
        action_pad_mask = action_pad_mask.unsqueeze(1).expand(-1, batch_size, -1)
        # Shape: (bsize * num_h, max_h)
        batched_action_pad_mask = action_pad_mask.permute(1, 0, 2).reshape(batch_size * num_horizons, self.max_horizon)

        # Embed the batched suffix inputs (state + action queries)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            batched_state, action_pad_mask=batched_action_pad_mask
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
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=suffix_att_2d_masks_4d,
            position_ids=suffix_position_ids,
            past_key_values=batched_past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],  # adarms_cond is None
        )

        # Project the output and extract the action predictions.
        predicted_actions_batched = self.action_out_proj(suffix_out.to(torch.float32))

        # Suffix structure: [state, action_query_1, ..., action_query_max]
        action_start_index = 1
        predicted_actions_padded = predicted_actions_batched[:,
                                   action_start_index: action_start_index + self.max_horizon, :]

        # Reshape the output to separate predictions for each head.
        # -> (bsize, num_h, max_h, dim) -> (num_h, bsize, max_h, dim)
        all_preds = predicted_actions_padded.view(
            batch_size, num_horizons, self.max_horizon, -1
        ).permute(1, 0, 2, 3)

        # Calculate losses in a parallelized manner.
        # 1. Primary Loss (L1): Ensures each expert head is trained well on its horizon.
        all_head_losses = []
        for i, h in enumerate(self.horizons):
            pred_head_i = all_preds[i, :, :h, :]
            target_actions_h = actions[:, :h, :]
            head_loss = F.l1_loss(pred_head_i, target_actions_h)  # L1 Loss
            all_head_losses.append(head_loss)

        individual_loss = torch.sum(torch.stack(all_head_losses))

        # 2. Auxiliary Loss (L1): Trains the gating network to combine predictions effectively.
        gate_logits = self.gate_out_proj(suffix_out.to(torch.float32))
        gate_logits = gate_logits[:, action_start_index: action_start_index + self.max_horizon,
                      :]  # B * num_horizons, max_horizon, 1

        # # Add Learnable Noise to gate logits
        # noise_epsilon = 1e-2
        # raw_noise_stddev = self.gate_noise_layer(suffix_out.to(torch.float32))
        # raw_noise_stddev = raw_noise_stddev[:, action_start_index: action_start_index + self.max_horizon, :]
        # noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
        # gate_logits = gate_logits + (torch.randn_like(gate_logits) * noise_stddev)

        gate_logits = gate_logits.reshape(batch_size, num_horizons, self.max_horizon).permute(0, 2, 1)

        valid_heads_mask = torch.tensor(
            [[step < h for h in self.horizons] for step in range(self.max_horizon)],
            device=actions.device,
            dtype=torch.bool
        ).unsqueeze(0)

        masked_gate_logits = torch.where(valid_heads_mask, gate_logits, torch.finfo(gate_logits.dtype).min)
        gate_weights = F.softmax(masked_gate_logits, dim=-1)

        # Combine predictions using the dynamic weights.
        # all_preds: (num_h, bsize, max_h, dim) -> (bsize, num_h, max_h, dim)
        all_preds_padded = all_preds.permute(1, 0, 2, 3)
        # combined_preds: (bsize, max_h, dim)
        combined_preds = (gate_weights.permute(0, 2, 1).unsqueeze(-1) * all_preds_padded).sum(dim=1)

        aux_loss_weight = loss_config.get("aux_weight", 1) if loss_config else 1
        auxiliary_loss = F.l1_loss(combined_preds, actions)  # L1 Loss

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
    def sample_actions(self, device, observation, ret_weights=False,
                       use_dynamic_replanning=False, l1_threshold=0.2,
                       min_replan_steps=1, max_replan_steps=20):
        """
        Do a single inference pass to predict actions directly using the gated ensemble.
        This replaces the multi-step diffusion sampling.
        """
        bsize = observation.state.shape[0]
        num_horizons = len(self.horizons)

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

        getattr(self.paligemma_with_expert, "gemma_expert_0").model.config._attn_implementation = "eager"

        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # STAGE 2: Batched Expert Suffix Pass (Single pass, no loop)
        # =============================================================
        batched_prefix_pad_masks = prefix_pad_masks.repeat_interleave(num_horizons, dim=0)
        batched_prefix_att_masks = prefix_att_masks.repeat_interleave(num_horizons, dim=0)
        batched_past_key_values = self._repeat_past_key_values(past_key_values, num_horizons)
        batched_past_key_values = DynamicCache.from_legacy_cache(past_key_values=batched_past_key_values)
        batched_state = state.repeat_interleave(num_horizons, dim=0)

        # Create batched action padding masks, one for each horizon
        action_pad_mask_list = []
        for h in self.horizons:
            pad_mask = F.pad(torch.ones((bsize, h), device=device, dtype=torch.bool),
                             (0, self.max_horizon - h),
                             value=False)
            action_pad_mask_list.append(pad_mask)
        action_pad_mask = torch.cat(action_pad_mask_list, dim=0)

        # Embed suffix (state + action queries)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            batched_state, action_pad_mask=action_pad_mask)

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

        # Single forward pass through all experts
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=suffix_att_2d_masks_4d,
            position_ids=suffix_position_ids,
            past_key_values=batched_past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],  # adarms_cond is None
        )

        # Suffix structure: [state, action_query_1, ..., action_query_max]
        action_start_index = 1

        # Get gate logits from the expert output
        gate_logits = self.gate_out_proj(suffix_out.to(torch.float32))
        gate_logits = gate_logits[:, action_start_index: action_start_index + self.max_horizon,
                      :]  # B * num_horizons, max_horizon, 1
        gate_logits = gate_logits.reshape(bsize, num_horizons, self.max_horizon).permute(0, 2, 1)

        valid_heads_mask = torch.tensor(
            [[step < h for h in self.horizons] for step in range(self.max_horizon)],
            device=device, dtype=torch.bool
        ).unsqueeze(0)

        masked_gate_logits = torch.where(valid_heads_mask, gate_logits, torch.finfo(gate_logits.dtype).min)
        gate_weights = F.softmax(masked_gate_logits, dim=-1)

        # Get action predictions from the expert output
        preds_batched = self.action_out_proj(suffix_out.to(torch.float32))

        # Slice out the action part from the full suffix output.
        preds_padded = preds_batched[:, action_start_index: action_start_index + self.max_horizon]

        # Reshape the result to align with (bsize, num_heads, max_horizon, action_dim)
        # (bsize * num_heads, max_horizon, dim) -> (bsize, num_heads, max_horizon, dim)
        all_individual_actions_padded = preds_padded.view(num_horizons, bsize, self.max_horizon, -1).permute(1, 0, 2, 3)

        # Combine predictions using the dynamic weights.
        fused_actions = (gate_weights.permute(0, 2, 1).unsqueeze(-1) * all_individual_actions_padded).sum(dim=1)

        # These are the predictions from each individual expert
        individual_actions = all_individual_actions_padded

        # STAGE 3: Dynamic Replanning (optional)
        # =============================================================
        return_dict = {}

        if use_dynamic_replanning:
            dynamic_replan_steps = min_replan_steps
            # Assuming bsize=1 for dynamic replanning
            fused_actions_squeeze = fused_actions[0]
            individual_actions_squeeze = individual_actions[0]

            calibration_distances = []
            for step in range(min_replan_steps):
                fused_action_s = fused_actions_squeeze[step]
                for h_idx, h in enumerate(self.horizons):
                    if step < h:  # Only consider active experts
                        individual_action_s_h = individual_actions_squeeze[h_idx, step]
                        dist = torch.linalg.norm((fused_action_s - individual_action_s_h), ord=1)
                        calibration_distances.append(dist)

            if not calibration_distances:
                dynamic_l1_threshold = l1_threshold
            else:
                all_dists = torch.stack(calibration_distances)
                dynamic_l1_threshold = torch.quantile(all_dists, 0.95)

            for step in range(min_replan_steps, self.max_horizon):
                fused_action_s = fused_actions_squeeze[step]
                votes_for_this_step = 0
                valid_horizons_for_this_step = 0

                for h_idx, h in enumerate(self.horizons):
                    if step < h:
                        valid_horizons_for_this_step += 1
                        individual_action_s_h = individual_actions_squeeze[h_idx, step]
                        l1_dist = torch.linalg.norm((fused_action_s - individual_action_s_h), ord=1)
                        if l1_dist < dynamic_l1_threshold:
                            votes_for_this_step += 1

                # Consensus requires a majority of active horizons to agree.
                if valid_horizons_for_this_step > 1 and votes_for_this_step > 2 * valid_horizons_for_this_step // 3:
                    dynamic_replan_steps += 1
                else:
                    break  # Break consensus on the first unstable step

            final_replan_steps = max(min_replan_steps, min(dynamic_replan_steps, max_replan_steps))
            return_dict["actions"] = fused_actions[:, :final_replan_steps, :]
            return_dict["replan_steps"] = final_replan_steps
        else:
            return_dict["actions"] = fused_actions

        if ret_weights:
            return_dict["gate_weights"] = gate_weights.detach().cpu()

        return return_dict