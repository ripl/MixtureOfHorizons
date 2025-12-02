import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override
from typing import List, Optional

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.models.pi0moh_config import Pi0GatedConfig


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


# Copied from pi0.py
@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)



class Pi0Gated(_model.BaseModel):

    def __init__(self, config: Pi0GatedConfig, rngs: nnx.Rngs):
        # Initialize base model with max_horizon.
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True] if config.pi05 else [False, False],
        )

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        # Attribute names must match pi0.py for weight loading.
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Shared action input projection.
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        if config.pi05:
            # Pi0.5-style: adaRMS conditioning on timestep.
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            # Pi0-style: state token + action-time MLP (no adaRMS).
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(
                2 * action_expert_config.width,
                action_expert_config.width,
                rngs=rngs,
            )
            self.action_time_mlp_out = nnx.Linear(
                action_expert_config.width,
                action_expert_config.width,
                rngs=rngs,
            )

        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Extra gating head for Mixture-of-Horizons.
        self.gate_out_proj = nnx.Linear(action_expert_config.width, 1, rngs=rngs)

    @at.typecheck
    def embed_prefix(
            self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
    ]:
        """Unchanged from pi0.py"""
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
            self, state,
            noisy_actions: _model.Actions,
            timestep: at.Float[at.Array, " b"],
            action_pad_mask,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """
        Pi0 / Pi0.5 compatible suffix embedding.

        Mirrors :class:`Pi0`'s ``embed_suffix`` (including adaRMS conditioning
        when ``pi05=True``) but takes ``state`` and ``action_pad_mask``
        explicitly to support batched horizon processing used by MoH.
        """
        input_mask = []
        ar_mask: list[bool] = []
        tokens = []

        adarms_cond = None

        # Optional Pi0-style state token (no state token for Pi0.5 / adaRMS).
        if not self.pi05:
            state_token = self.state_proj(state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        # Timestep embedding.
        time_emb = posemb_sincos(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )

        # Project actions.
        action_tokens = self.action_in_proj(noisy_actions)

        if self.pi05:
            # Pi0.5: adaRMS on time embedding, actions unchanged.
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            adarms_cond = time_emb
            action_expert_tokens = action_tokens
        else:
            # Pi0: concatenate time + actions and pass through MLP.
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=noisy_actions.shape[1])
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens

        if action_pad_mask is None:
            action_pad_mask = jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_)
        input_mask.append(action_pad_mask)

        tokens.append(action_expert_tokens)

        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (action_expert_tokens.shape[1] - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask_arr = jnp.concatenate(input_mask, axis=1)
        ar_mask_arr = jnp.array(ar_mask)

        return tokens, input_mask_arr, ar_mask_arr, adarms_cond

    def cv_squared(self, x: at.Array, eps: float = 1e-10) -> at.Array:
        """Computes the squared coefficient of variation. (From pi0_pytorch_moh.py)"""

        def compute_cv():
            mean = jnp.mean(x, dtype=jnp.float32)
            var = jnp.var(x, dtype=jnp.float32)
            return var / (jnp.square(mean) + eps)

        # Handle num_experts = 1 case
        return jax.lax.cond(
            x.shape[0] == 1,
            lambda: jnp.array(0.0, dtype=jnp.float32),
            compute_cv
        )

    @override
    def compute_loss(
            self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
    # def compute_loss(
    #     self,
    #     rng: at.KeyArrayLike,
    #     observation: _model.Observation,
    #     actions: at.Float[at.Array, "b s action_dim"],
    # ) -> tuple[at.Float[at.Array, ""], dict[str, at.Float[at.Array, ""]]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_size, max_horizon, action_dim = actions.shape
        num_horizons = len(self.config.horizons)
        horizons_arr = jnp.array(self.config.horizons)

        # Sample noise and time
        noise = jax.random.normal(noise_rng, actions.shape)
        time_scalar = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001

        # Expand time and actions for each horizon
        # time shape: (H, B)
        time = einops.repeat(time_scalar, "b -> h b", h=num_horizons)
        # x_t shape: (H, B, max_H, D)
        x_t = time[..., None, None] * noise[None, ...] + (1 - time[..., None, None]) * actions[None, ...]
        # u_t (target) shape: (B, max_H, D)
        u_t = noise - actions

        # STAGE 1: VLM Prefix Pass (Compute KV cache once)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (_, prefix_out), prefix_past_key_values = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions
        )

        # STAGE 2: Action Head Suffix Passes (Parallelized via batching)

        # Repeat prefix masks and KV cache for each horizon
        # New batch size is (B * H)
        batched_prefix_mask = jnp.repeat(prefix_mask, num_horizons, axis=0)
        batched_past_key_values = jax.tree_map(
            lambda x: jnp.repeat(x, num_horizons, axis=1),
            prefix_past_key_values
        )
        batched_state = jnp.repeat(observation.state, num_horizons, axis=0)

        # Reshape x_t and time to align with the new batch dimension
        # (H, B, max_H, D) -> (B*H, max_H, D)
        batched_x_t = jnp.transpose(x_t, (1, 0, 2, 3)).reshape(batch_size * num_horizons, max_horizon, -1)
        # (H, B) -> (B*H,)
        batched_time = jnp.transpose(time, (1, 0)).reshape(-1)

        # Create a padding mask for actions based on valid horizon length
        # (H, max_H)
        action_pad_mask = jnp.arange(max_horizon)[None, :] < horizons_arr[:, None]
        # (B*H, max_H)
        action_pad_mask_expanded = jnp.broadcast_to(
            action_pad_mask[None, :, :],
            (batch_size, num_horizons, max_horizon)
        )
        # (B, H, max_H) -> (B*H, max_H)
        batched_action_pad_mask = action_pad_mask_expanded.reshape(batch_size * num_horizons, max_horizon)

        # Embed the batched suffix inputs
        suffix_tokens, suffix_pad_masks, suffix_ar_mask, adarms_cond = self.embed_suffix(
            batched_state, batched_x_t, batched_time, action_pad_mask=batched_action_pad_mask
        )

        # Combine prefix and suffix masks for cross-attention
        pad_masks = jnp.concatenate([batched_prefix_mask, suffix_pad_masks], axis=1)
        ar_masks = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        full_att_2d_masks = make_attn_mask(pad_masks, ar_masks)

        prefix_len = prefix_mask.shape[1]
        suffix_len = suffix_tokens.shape[1]

        # Create position IDs and attention mask for the suffix part only
        suffix_position_ids = jnp.arange(prefix_len, prefix_len + suffix_len)[None, :]
        suffix_att_2d_masks = full_att_2d_masks[:, -suffix_len:, :]

        b = suffix_att_2d_masks.shape[0]
        suffix_position_ids = jnp.broadcast_to(suffix_position_ids, (b, suffix_len))

        adarms = [None, adarms_cond] if self.pi05 else [None, None]
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=suffix_att_2d_masks,
            positions=suffix_position_ids,
            kv_cache=batched_past_key_values,
            adarms_cond=adarms,
        )
        
        action_start_index = 0 if self.pi05 else 1  # pi0.5 has no state token
        v_t_batched = self.action_out_proj(suffix_out)
        v_t_actions_padded = v_t_batched[:, action_start_index: action_start_index + max_horizon, :]
        # (H, B, max_H, D_action)
        all_v_t_preds = v_t_actions_padded.reshape(
            batch_size, num_horizons, max_horizon, -1
        ).transpose(1, 0, 2, 3)

        # 1. Primary Loss: Ensures each expert head is trained well.
        all_head_losses = []
        for i, h in enumerate(self.config.horizons):
            v_t_head = all_v_t_preds[i, :, :h, :]
            target_v_t = u_t[:, :h, :]
            # Mean over batch, horizon, and action dim
            head_loss = jnp.mean(jnp.square(v_t_head - target_v_t))
            all_head_losses.append(head_loss)

        individual_loss = jnp.sum(jnp.stack(all_head_losses))

        # 2. Auxiliary Loss: Trains the gating network
        # (B*H, S_suffix, 1)
        gate_logits_batched = self.gate_out_proj(suffix_out)
        # (B*H, max_H, 1)
        gate_logits_padded = gate_logits_batched[:, action_start_index: action_start_index + max_horizon, :]
        # (B, max_H, H)
        # gate_logits = einops.rearrange(gate_logits_padded, "(b h) s 1 -> b s h", b=batch_size, h=num_horizons)
        gate_logits_reshaped = gate_logits_padded.reshape(batch_size, num_horizons, max_horizon, 1)
        gate_logits = jnp.transpose(gate_logits_reshaped, (0, 2, 1, 3)).squeeze(-1)

        # Create mask for softmax
        # (max_H, H)
        valid_heads_mask = jnp.arange(max_horizon)[:, None] < horizons_arr[None, :]
        # (B, max_H, H) - broadcast batch dim
        valid_heads_mask = jnp.broadcast_to(valid_heads_mask, gate_logits.shape)

        masked_gate_logits = jnp.where(valid_heads_mask, gate_logits, jnp.finfo(gate_logits.dtype).min)
        gate_weights = nnx.softmax(masked_gate_logits, axis=-1)

        # Combine predictions using the dynamic weights
        # all_v_t_preds: (H, B, max_H, D) -> (B, H, max_H, D)
        all_v_t_preds_permuted = einops.rearrange(all_v_t_preds, "h b s d -> b h s d")
        # gate_weights: (B, max_H, H) -> (B, H, max_H, 1)
        gate_weights_expanded = jnp.transpose(gate_weights, (0, 2, 1))[:, :, :, None]

        # (B, H, max_H, D) * (B, H, max_H, 1) -> sum over H -> (B, max_H, D)
        v_t_combined = jnp.sum(all_v_t_preds_permuted * gate_weights_expanded, axis=1)

        auxiliary_loss = jnp.mean(jnp.square(v_t_combined - u_t))  # Mean over B, H, D

        # 3. Balance Loss: Encourage the gate layer to output weights flexibly
        loss_components = []
        boundaries = sorted(list(set([0] + self.config.horizons)))
        for i in range(len(boundaries) - 1):
            start_step, end_step = boundaries[i], boundaries[i + 1]
            active_expert_indices = [idx for idx, h in enumerate(self.config.horizons) if h > start_step]

            if len(active_expert_indices) > 1:
                # (B, S_segment, H_total)
                segment_gate_weights = gate_weights[:, start_step:end_step, :]
                # (B, S_segment, H_active)
                active_expert_weights = segment_gate_weights[:, :, jnp.array(active_expert_indices)]
                # (H_active,)
                avg_expert_prob_in_segment = jnp.mean(active_expert_weights, axis=(0, 1))
                segment_loss = self.cv_squared(avg_expert_prob_in_segment)
                loss_components.append(segment_loss)

        load_balancing_loss = jnp.mean(jnp.stack(loss_components)) if loss_components else 0.0

        total_loss = (
                individual_loss +
                self.config.aux_weight * auxiliary_loss +
                self.config.balance_weight * load_balancing_loss
        )

        return total_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """
        Samples actions using the gated fusion mechanism during denoising.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        max_horizon = self.action_horizon
        num_horizons = len(self.config.horizons)
        horizons_arr = jnp.array(self.config.horizons)

        noise = jax.random.normal(rng, (batch_size, max_horizon, self.action_dim))

        # First fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (_, prefix_out), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions
        )

        # Prepare static batched inputs (these don't change in the loop)
        batched_prefix_mask = jnp.repeat(prefix_mask, num_horizons, axis=0)
        batched_kv_cache = jax.tree_map(
            lambda x: jnp.repeat(x, num_horizons, axis=1),
            kv_cache
        )
        batched_state = jnp.repeat(observation.state, num_horizons, axis=0)

        # Create static masks for padding actions in the loop
        # (H, max_H)
        steps_arr = jnp.arange(max_horizon)
        action_pad_mask_per_h = steps_arr[None, :] < horizons_arr[:, None]
        # (B*H, max_H)
        batched_action_pad_mask = jnp.broadcast_to(
            action_pad_mask_per_h[None, :, :],
            (batch_size, num_horizons, max_horizon)
        )
        batched_action_pad_mask = einops.rearrange(batched_action_pad_mask, "b h s -> (b h) s")
        # batched_action_pad_mask = einops.repeat(action_pad_mask_per_h, "h s -> (b h) s", b=batch_size)
        # (H, max_H, 1)
        action_mask_for_padding_x_t = (steps_arr[None, :, None] < horizons_arr[:, None, None])

        # Create static mask for gate softmax
        # (max_H, H)
        valid_heads_mask = steps_arr[:, None] < horizons_arr[None, :]
        # (B, max_H, H) - for broadcasting
        valid_heads_mask = valid_heads_mask[None, :, :]
        
        action_start_index = 0 if self.pi05 else 1  # pi0.5 has no state token

        prefix_len = prefix_mask.shape[1]

        def step_fn(carry):
            x_t, time = carry

            # --- Prepare Batched Inputs for this step ---
            expanded_time = jnp.broadcast_to(time, (batch_size * num_horizons,))

            # Pad x_t for each horizon
            # (1, B, max_H, D)
            x_t_expanded = x_t[None, ...]
            # (H, B, max_H, D)
            padded_x_t_batched = jnp.where(action_mask_for_padding_x_t, x_t_expanded, 0.0)
            # (B*H, max_H, D)
            batched_x_t = padded_x_t_batched.transpose(1, 0, 2, 3)
            batched_x_t = einops.rearrange(batched_x_t, "b h s d -> (b h) s d")

            # --- Run Batched Suffix Pass ---
            suffix_tokens, suffix_pad_masks, suffix_ar_mask, adarms_cond = self.embed_suffix(
                batched_state, batched_x_t, expanded_time, action_pad_mask=batched_action_pad_mask
            )

            pad_masks = jnp.concatenate([batched_prefix_mask, suffix_pad_masks], axis=1)
            ar_masks = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            full_att_2d_masks = make_attn_mask(pad_masks, ar_masks)

            suffix_len = suffix_tokens.shape[1]
            suffix_position_ids = jnp.arange(prefix_len, prefix_len + suffix_len)[None, :]
            suffix_att_2d_masks = full_att_2d_masks[:, -suffix_len:, :]

            b = suffix_att_2d_masks.shape[0]
            suffix_position_ids = jnp.broadcast_to(suffix_position_ids, (b, suffix_len))
            
            adarms = [None, adarms_cond] if self.pi05 else [None, None]
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=suffix_att_2d_masks,
                positions=suffix_position_ids,
                kv_cache=batched_kv_cache,
                adarms_cond=adarms,
            )

            # --- Gating and Fusion ---
            # (B*H, S_suffix, 1)
            gate_logits_batched = self.gate_out_proj(suffix_out)
            # (B*H, max_H, 1)
            gate_logits_padded = gate_logits_batched[:, action_start_index: action_start_index + max_horizon, :]
            # (B, max_H, H)
            gate_logits_reshaped = gate_logits_padded.reshape(batch_size, num_horizons, max_horizon, 1)
            gate_logits = jnp.transpose(gate_logits_reshaped, (0, 2, 1, 3)).squeeze(-1)
            masked_gate_logits = jnp.where(valid_heads_mask, gate_logits, jnp.finfo(gate_logits.dtype).min)
            gate_weights = nnx.softmax(masked_gate_logits, axis=-1)

            # Get all predictions
            # (B*H, S_suffix, D_action)
            v_t_batched = self.action_out_proj(suffix_out)
            # (B*H, max_H, D_action)
            v_t_actions_padded = v_t_batched[:, action_start_index: action_start_index + max_horizon, :]
            # (B, H, max_H, D_action)
            all_v_t_preds = v_t_actions_padded.reshape(batch_size, num_horizons, max_horizon, -1)

            # Combine predictions
            # gate_weights: (B, max_H, H) -> (B, H, max_H, 1)
            gate_weights_expanded = jnp.transpose(gate_weights, (0, 2, 1))[:, :, :, None]

            # (B, H, max_H, D) * (B, H, max_H, 1) -> sum over H -> (B, max_H, D)
            v_t = jnp.sum(all_v_t_preds * gate_weights_expanded, axis=1)

            # --- Euler Step ---
            x_t_new = x_t + dt * v_t
            time_new = time + dt

            return (x_t_new, time_new)

        def cond_fn(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond_fn, step_fn, (noise, 1.0))
        return x_0

