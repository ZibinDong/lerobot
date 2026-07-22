# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

"""Complete G0.5 model and LeRobot policy entry points."""

from __future__ import annotations

import copy
from collections import deque
from typing import TYPE_CHECKING

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available

from .configuration_g05 import G05Config
from .modular_g05 import (
    G05_ACTION_DIM_PAD_MASK,
    G05_ACTION_PAD_MASK,
    G05_ATTENTION_MASK,
    G05_IMAGES,
    G05_INPUT_IDS,
    G05_LABELS,
    G05_PREFIX_LENGTH,
    ActionExpert,
    G05Attention,
    G05GatedDeltaNet,
    G05VisionModel,
)

if TYPE_CHECKING or _transformers_available:
    from transformers import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel


class G05Model(nn.Module):
    """Qwen3.5 VLM plus the flow-matching G0.5 action expert."""

    def __init__(self, config: G05Config) -> None:
        super().__init__()
        text_config = Qwen3_5TextConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.text_hidden_size,
            intermediate_size=config.text_intermediate_size,
            num_hidden_layers=config.text_num_layers,
            num_attention_heads=config.text_num_heads,
            num_key_value_heads=config.text_num_kv_heads,
            head_dim=config.text_head_dim,
            max_position_embeddings=262_144,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": 0.25,
                "mrope_section": list(config.mrope_section),
                "mrope_interleaved": True,
            },
            layer_types=config.text_layer_types,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
        )
        vision_config = Qwen3_5VisionConfig(
            depth=config.vision_depth,
            hidden_size=config.vision_hidden_size,
            intermediate_size=config.vision_intermediate_size,
            num_heads=config.vision_num_heads,
            patch_size=config.vision_patch_size,
            temporal_patch_size=config.vision_temporal_patch_size,
            spatial_merge_size=config.vision_spatial_merge_size,
            out_hidden_size=config.text_hidden_size,
        )
        # G0.5 full-attention layers use the explicit eager causal mask. This
        # must be selected before TextModel builds its mask interface; SDPA may
        # otherwise elide the mask and rely on an internal ``is_causal`` flag.
        text_config._attn_implementation = "eager"
        self.config = config
        self.vision_tower = G05VisionModel(vision_config)
        self.vlm = Qwen3_5TextModel(text_config)
        # Transformers' generic Qwen3.5 linear attention uses different chunk
        # and precision defaults. Replace only those layers while retaining the
        # same checkpoint parameter paths and DynamicCache contract.
        for index, layer in enumerate(self.vlm.layers):
            if layer.layer_type == "linear_attention":
                layer.linear_attn = G05GatedDeltaNet(text_config, index)
            else:
                layer.self_attn = G05Attention(text_config, index)
        self.output_proj = nn.Linear(config.text_hidden_size, config.vocab_size, bias=False)
        self.output_proj.weight = self.vlm.embed_tokens.weight
        self.proprio_embedder = nn.Sequential(
            nn.Linear(config.internal_state_dim, config.text_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.text_hidden_size),
            nn.Linear(config.text_hidden_size, config.text_hidden_size),
        )
        self.action_expert = ActionExpert(config)

    def encode_images(self, images: Tensor) -> Tensor:
        """Patchify all cameras/timesteps and encode them as one packed vision batch.

        G0.5 repeats each single frame over the temporal patch axis. The einops
        layout mirrors the official token order: image, merged-grid row/column,
        merge row/column, then flattened channel/temporal/patch content.
        """
        if images.ndim == 5:
            images = rearrange(
                images, "batch camera channel height width -> batch camera 1 channel height width"
            )
        if images.ndim != 6:
            raise ValueError("g05_images must have shape [B,Cam,Time,C,H,W]")

        batch_size, num_cameras, num_frames, _, height, width = images.shape
        patch_size = self.config.vision_patch_size
        merge_size = self.config.vision_spatial_merge_size
        temporal_patch = self.config.vision_temporal_patch_size
        grid_height, grid_width = height // patch_size, width // patch_size
        frames = repeat(
            images,
            "batch camera time channel height width -> (batch camera time) channel temporal height width",
            temporal=temporal_patch,
        )
        # The official checkpoint was trained with a legacy channel/temporal
        # reinterpretation before patch extraction. These two named layouts
        # preserve that exact order without an opaque reshape/permute chain.
        legacy_frames = rearrange(
            frames, "frames channel temporal height width -> frames (channel temporal) height width"
        )
        patches = rearrange(
            legacy_frames,
            "frames (temporal channel) "
            "(grid_h merge_h patch_h) (grid_w merge_w patch_w) -> "
            "(frames grid_h grid_w merge_h merge_w) (channel temporal patch_h patch_w)",
            temporal=temporal_patch,
            channel=images.shape[3],
            merge_h=merge_size,
            merge_w=merge_size,
            patch_h=patch_size,
            patch_w=patch_size,
        )

        num_images = batch_size * num_cameras * num_frames
        grid_thw = torch.tensor((1, grid_height, grid_width), device=images.device, dtype=torch.long).expand(
            num_images, -1
        )
        encoded = self.vision_tower(patches, grid_thw).pooler_output
        return rearrange(encoded, "(batch tokens) hidden -> batch tokens hidden", batch=batch_size)

    def build_mrope_position_ids(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Build vectorized Qwen3.5 multimodal rotary positions.

        Text advances all three axes by one. An image block uses a constant
        temporal coordinate and a 2-D spatial grid, then advances the following
        text by ``max(grid_height, grid_width)``. Left-padding stays at position 0.
        Processor-generated prompts guarantee contiguous, fixed-size image blocks.
        """
        merged_height = self.config.image_size[0] // (
            self.config.vision_patch_size * self.config.vision_spatial_merge_size
        )
        merged_width = self.config.image_size[1] // (
            self.config.vision_patch_size * self.config.vision_spatial_merge_size
        )
        image_length = merged_height * merged_width
        image_span = max(merged_height, merged_width)
        image_mask = input_ids.eq(self.config.image_token_id) & attention_mask
        if torch.any(image_mask.sum(dim=-1).remainder(image_length)):
            raise ValueError(f"every image must occupy exactly {image_length} prompt tokens")

        valid_rank = attention_mask.long().cumsum(dim=-1) - 1
        image_token_rank = image_mask.long().cumsum(dim=-1) - 1
        within_image = image_token_rank.remainder(image_length)
        image_number = torch.div(image_token_rank, image_length, rounding_mode="floor")
        completed_images = torch.div(image_mask.long().cumsum(dim=-1), image_length, rounding_mode="floor")

        text_position = valid_rank - completed_images * (image_length - image_span)
        image_position = valid_rank - within_image - image_number * (image_length - image_span)
        temporal = torch.where(image_mask, image_position, text_position)
        height = torch.where(
            image_mask,
            image_position + torch.div(within_image, merged_width, rounding_mode="floor"),
            text_position,
        )
        width = torch.where(
            image_mask,
            image_position + within_image.remainder(merged_width),
            text_position,
        )
        position_ids = torch.stack((temporal, height, width))
        return position_ids.masked_fill(~attention_mask.unsqueeze(0), 0)

    def vlm_forward(self, batch: dict[str, Tensor], *, use_cache: bool):
        """Run the multimodal language model for either prefix caching or AR loss."""
        input_ids = batch[G05_INPUT_IDS]
        attention_mask = batch[G05_ATTENTION_MASK].bool()
        inputs_embeds = self.vlm.embed_tokens(input_ids)

        image_features = self.encode_images(batch[G05_IMAGES]).to(inputs_embeds.dtype)
        image_mask = input_ids.eq(self.config.image_token_id)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(-1), image_features)

        if self.config.state_token_id is not None:
            state_mask = input_ids.eq(self.config.state_token_id)
            # The released model keeps the proprio encoder in fp32. A single
            # bf16-rounded state token changes GatedDeltaNet's recurrent state
            # and consequently every token that follows it.
            with torch.autocast(inputs_embeds.device.type, enabled=False):
                state_features = self.proprio_embedder(batch[OBS_STATE].float())
            state_features = state_features.to(inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(state_mask.unsqueeze(-1), state_features)

        position_ids = self.build_mrope_position_ids(input_ids, attention_mask)
        outputs = self.vlm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            return_dict=True,
        )
        return outputs, position_ids, attention_mask

    def prefill(self, batch: dict[str, Tensor]) -> tuple[DynamicCache, Tensor, Tensor, Tensor]:
        """Encode only the prefix, excluding teacher-forced action tokens."""
        prefix_length = int(batch.get(G05_PREFIX_LENGTH, batch[G05_INPUT_IDS].shape[1]))
        prefix_batch = {
            **batch,
            G05_INPUT_IDS: batch[G05_INPUT_IDS][:, :prefix_length],
            G05_ATTENTION_MASK: batch[G05_ATTENTION_MASK][:, :prefix_length],
        }
        outputs, position_ids, attention_mask = self.vlm_forward(prefix_batch, use_cache=True)
        return outputs.past_key_values, position_ids, attention_mask, outputs.last_hidden_state[:, -1:]

    def copy_vlm_cache(self, source: DynamicCache) -> DynamicCache:
        """Clone mutable linear-attention state while sharing immutable prefix KV."""
        destination = DynamicCache(config=self.vlm.config)
        for index, source_layer in enumerate(source.layers):
            destination_layer = copy.copy(source_layer)
            if not hasattr(source_layer, "is_initialized"):
                destination_layer.conv_states = source_layer.conv_states.clone()
                destination_layer.recurrent_states = source_layer.recurrent_states.clone()
            destination.layers[index] = destination_layer
        return destination

    def autoregressive_loss(
        self,
        batch: dict[str, Tensor],
        prefill: tuple[DynamicCache, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute shifted CE and action/CoT accuracies over labeled suffix tokens."""
        if G05_LABELS not in batch:
            raise ValueError("discrete G0.5 training requires action-token labels from the processor")
        prefix_cache, _, _, prefix_last_hidden = prefill
        suffix_cache = self.copy_vlm_cache(prefix_cache)
        prefix_length = int(batch[G05_PREFIX_LENGTH])
        suffix_ids = batch[G05_INPUT_IDS][:, prefix_length:]
        suffix_positions = self.build_mrope_position_ids(
            batch[G05_INPUT_IDS], batch[G05_ATTENTION_MASK].bool()
        )[:, :, prefix_length:]
        outputs = self.vlm(
            inputs_embeds=self.vlm.embed_tokens(suffix_ids),
            attention_mask=batch[G05_ATTENTION_MASK].bool(),
            position_ids=suffix_positions,
            past_key_values=suffix_cache,
            use_cache=False,
            return_dict=True,
        )
        # The final prefix state predicts the first action token; each suffix
        # state then predicts the following action token or EOS.
        hidden_states = torch.cat((prefix_last_hidden, outputs.last_hidden_state[:, :-1]), dim=1)
        labels = batch[G05_LABELS][:, prefix_length:]
        valid = labels.ne(-100)
        if not valid.any():
            zero = hidden_states.sum() * 0
            return zero, zero.detach(), zero.detach()
        hidden_states = hidden_states[valid]
        targets = labels[valid]
        logits = self.output_proj(hidden_states)
        loss = torch.nn.functional.cross_entropy(logits.float(), targets)
        correct = logits.detach().argmax(-1).eq(targets)
        start, end = self.config.action_token_start_id, self.config.action_token_end_id
        if start is None or end is None:
            raise ValueError("converted discrete G0.5 config is missing its action-token range")
        action_mask = targets.ge(start) & targets.lt(end)
        action_accuracy = correct[action_mask].float().mean() if action_mask.any() else loss.new_zeros(())
        cot_mask = ~action_mask
        cot_accuracy = correct[cot_mask].float().mean() if cot_mask.any() else loss.new_zeros(())
        return loss, action_accuracy, cot_accuracy

    def flow_loss(
        self,
        batch: dict[str, Tensor],
        prefill: tuple[DynamicCache, Tensor, Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Compute the masked conditional-flow velocity objective."""
        actions = batch[ACTION]
        prefix_cache, prefix_position_ids, prefix_attention_mask, _ = prefill or self.prefill(batch)
        if not self.config.flow_joint_training:
            # Post-training variants optimize only the expert; detaching cached KV
            # avoids retaining the large VLM backward graph.
            for layer in prefix_cache.layers:
                if layer.is_initialized:
                    layer.keys = layer.keys.detach()
                    layer.values = layer.values.detach()

        num_samples = self.config.num_flow_samples
        batch_size = actions.shape[0]
        if self.config.flow_sampling == "beta":
            # Official training samples Beta times on CPU. One batched transfer
            # preserves its seeded RNG sequence without per-sample transfers.
            distribution = torch.distributions.Beta(self.config.flow_beta_alpha, self.config.flow_beta_beta)
            time = distribution.sample((num_samples, batch_size)).to(
                device=actions.device, dtype=actions.dtype
            )
            time = 1 - (1 - self.config.flow_sig_min) * (1 - time)
        else:
            offsets = torch.arange(batch_size, device=actions.device, dtype=actions.dtype)
            time = (
                torch.rand(num_samples, 1, device=actions.device, dtype=actions.dtype) + offsets / batch_size
            ).remainder(1 - 1e-5)

        time = rearrange(time, "samples batch -> (samples batch)")
        noise = torch.randn(
            num_samples,
            *actions.shape,
            device=actions.device,
            dtype=actions.dtype,
        )
        noise = rearrange(noise, "samples batch horizon dim -> (samples batch) horizon dim")
        actions = repeat(actions, "batch horizon dim -> (samples batch) horizon dim", samples=num_samples)
        noised_actions = (1 - time[:, None, None]) * actions + time[:, None, None] * noise

        if num_samples > 1:
            prefix_cache = self.action_expert.copy_prefix_cache(prefix_cache)
            prefix_cache.batch_repeat_interleave(num_samples)
            prefix_position_ids = repeat(
                prefix_position_ids,
                "axes batch tokens -> axes (samples batch) tokens",
                samples=num_samples,
            )
            prefix_attention_mask = repeat(
                prefix_attention_mask,
                "batch tokens -> (samples batch) tokens",
                samples=num_samples,
            )

        velocity = self.action_expert(
            noised_actions,
            time,
            prefix_cache,
            prefix_position_ids,
            prefix_attention_mask,
        )
        squared_error = (velocity - (noise - actions)).square()
        weights = torch.ones_like(squared_error)
        if G05_ACTION_PAD_MASK in batch:
            action_pad_mask = repeat(
                batch[G05_ACTION_PAD_MASK],
                "batch horizon -> (samples batch) horizon",
                samples=num_samples,
            )
            weights.masked_fill_(action_pad_mask[..., None], 0)
        if G05_ACTION_DIM_PAD_MASK in batch:
            dimension_pad_mask = repeat(
                batch[G05_ACTION_DIM_PAD_MASK],
                "batch dim -> (samples batch) dim",
                samples=num_samples,
            )
            weights.masked_fill_(dimension_pad_mask[:, None, :], 0)
        return (weights * squared_error).sum() / weights.sum().clamp_min(1)

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Integrate the learned velocity field from Gaussian noise to actions."""
        prefix_cache, prefix_position_ids, prefix_attention_mask, _ = self.prefill(batch)
        batch_size = batch[G05_INPUT_IDS].shape[0]
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.config.chunk_size,
                self.config.internal_action_dim,
                device=batch[G05_INPUT_IDS].device,
                dtype=batch[OBS_STATE].dtype,
            )
        dimension_pad_mask = batch.get(G05_ACTION_DIM_PAD_MASK)
        if dimension_pad_mask is not None:
            noise = noise.masked_fill(dimension_pad_mask[:, None, :], 0)
        actions = noise
        step_size = 1 / self.config.num_inference_steps
        time = torch.ones(batch_size, device=actions.device, dtype=actions.dtype)
        for _ in range(self.config.num_inference_steps):
            velocity = self.action_expert(
                actions,
                time,
                prefix_cache,
                prefix_position_ids,
                prefix_attention_mask,
            )
            actions = actions - step_size * velocity
            if dimension_pad_mask is not None:
                actions.masked_fill_(dimension_pad_mask[:, None, :], 0)
            time = time - step_size
        return actions


class G05Policy(PreTrainedPolicy):
    """LeRobot policy wrapper for training and chunked G0.5 inference."""

    config_class = G05Config
    name = "g05"

    def __init__(self, config: G05Config, **kwargs) -> None:
        super().__init__(config)
        if not _transformers_available:
            raise ImportError("G0.5 requires transformers; install lerobot[g05]")
        config.validate_features()
        self.model = G05Model(config)
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)
        self._physical_action_dim = config.output_features[ACTION].shape[-1]

    def reset(self) -> None:
        self._action_queue.clear()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the configured continuous-flow and discrete-token objectives."""
        device_type = batch[G05_INPUT_IDS].device.type
        with torch.autocast(
            device_type,
            dtype=torch.bfloat16,
            enabled=self.config.dtype == "bfloat16" and device_type == "cuda",
        ):
            prefill = self.model.prefill(batch)
            if self.config.discrete_action:
                ar_loss, action_accuracy, cot_accuracy = self.model.autoregressive_loss(batch, prefill)
                ar_loss = ar_loss * self.config.action_token_loss_weight
            else:
                ar_loss = prefill[3].new_zeros(())
                action_accuracy = cot_accuracy = ar_loss.detach()
            fm_loss = self.model.flow_loss(batch, prefill=prefill) * self.config.fm_loss_weight
        loss = fm_loss + ar_loss
        return loss, {
            "fm_loss": fm_loss.detach(),
            "action_token_loss": ar_loss.detach(),
            "cot_loss": ar_loss.detach().new_zeros(()),
            "action_token_accuracy": action_accuracy,
            "cot_accuracy": cot_accuracy if self.config.predict_cot else cot_accuracy.new_zeros(()),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a physical action chunk, removing checkpoint-only padded dimensions."""
        if self.config.action_attend_cot:
            raise NotImplementedError(
                "this checkpoint requires autoregressive CoT generation before flow inference"
            )
        device_type = batch[G05_INPUT_IDS].device.type
        with torch.autocast(
            device_type,
            dtype=torch.bfloat16,
            enabled=self.config.dtype == "bfloat16" and device_type == "cuda",
        ):
            actions = self.model.sample_actions(batch, noise=noise)
        indices = self.config.action_indices or list(range(self._physical_action_dim))
        return actions[..., indices]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Serve one action while amortizing model inference over a predicted chunk."""
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch, noise=noise)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()
