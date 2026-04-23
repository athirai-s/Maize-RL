from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Callable, Any, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree
from transformers.generation import GenerationConfig
from transformers.modeling_flax_outputs import FlaxCausalLMOutputWithCrossAttentions
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
from flax.training.train_state import TrainState
import flax.linen as nn
from optax import softmax_cross_entropy_with_integer_labels
from jax.experimental.pjit import pjit
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS

from JaxSeq.utils import BlockingStrategy, Padding, Truncation, strip_prompt_from_completion
from JaxSeq.utils import with_named_sharding_constraint, match_partition_rules, block_sequences, multihost_device_get
from JaxSeq.models.base_interface import initialize_attn_mask_pos_ids
from LLM_RL.environment import Text, TextHistory, text_history_to_str, TokenTrajectoryChain, TextTrajectoryChain
from LLM_RL.algorithms.ppo.base_interface import (
    PPOTrain, PPOInference, PPOPolicy,
    CombinedTokenTrajectoryChain, get_action_state_next_state_idxs,
    get_advantages_and_returns, whiten,
)
from LLM_RL.algorithms.ppo.data_multimodal import PPOData
from llm_rl_scripts.maze.model.local_patch_encoder import LocalPatchCNN


def _extract_last_patch_from_history(text_history: TextHistory, patch_size: int) -> np.ndarray:
    for item in reversed(text_history):
        if (not item.is_action) and getattr(item, 'extras', None) is not None and 'local_patch' in item.extras:
            return np.asarray(item.extras['local_patch'], dtype=np.int32)
    patch = np.ones((patch_size, patch_size), dtype=np.int32)
    patch[patch_size // 2, patch_size // 2] = 2
    return patch


def _text_chain_to_list(chain: TextTrajectoryChain):
    out = []
    curr = chain
    while curr is not None:
        out.append(curr.text_trajectory)
        curr = curr.next
    return out


def _extract_chain_patches(text_trajectory_chains: List[TextTrajectoryChain], patch_size: int) -> List[np.ndarray]:
    patches = []
    for chain in text_trajectory_chains:
        for tt in _text_chain_to_list(chain):
            state = tt.text_history[0]
            extras = getattr(state, 'extras', None)
            if extras is not None and 'local_patch' in extras:
                patches.append(np.asarray(extras['local_patch'], dtype=np.int32))
            else:
                patch = np.ones((patch_size, patch_size), dtype=np.int32)
                patch[patch_size // 2, patch_size // 2] = 2
                patches.append(patch)
    return patches


def _get_embedding_table(params: PyTree) -> jnp.ndarray:
    try:
        return params['transformer']['wte']['embedding']
    except Exception:
        return params['wte']['embedding']


def _build_inputs_embeds(model: FlaxPreTrainedModel, params: PyTree, input_ids: jax.Array, local_patch: Optional[jax.Array], encoder_module: LocalPatchCNN, encoder_params: PyTree):
    token_emb = _get_embedding_table(params)
    text_embeds = jnp.take(token_emb, input_ids, axis=0)
    prefix_len = 0
    if local_patch is None:
        return text_embeds, prefix_len
    visual_embeds = encoder_module.apply(encoder_params, local_patch.astype(jnp.int32), train=False)
    prefix_len = visual_embeds.shape[1]
    return jnp.concatenate([visual_embeds, text_embeds], axis=1), prefix_len


def _lm_forward_with_embeds(module, input_embeds, attention_mask, position_ids, deterministic, output_attentions, output_hidden_states):
    transformer = module.transformer
    pos_embeds = transformer.wpe(position_ids.astype("i4"))
    hidden_states = input_embeds + pos_embeds
    hidden_states = transformer.dropout(hidden_states, deterministic=deterministic)
    outputs = transformer.h(
        hidden_states,
        attention_mask,
        None,
        None,
        deterministic=deterministic,
        init_cache=False,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    hidden_states = outputs[0]
    hidden_states = transformer.ln_f(hidden_states)
    if output_hidden_states:
        all_hidden_states = outputs[1] + (hidden_states,)
    else:
        all_hidden_states = None
    if module.config.tie_word_embeddings:
        shared_kernel = transformer.variables["params"]["wte"]["embedding"].T
        lm_logits = module.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
    else:
        lm_logits = module.lm_head(hidden_states)
    return FlaxCausalLMOutputWithCrossAttentions(
        logits=lm_logits,
        hidden_states=all_hidden_states,
        attentions=outputs[2] if output_attentions else None,
        cross_attentions=None,
    )


def _run_model(model, params, input_ids, attention_mask, local_patch, encoder_module, encoder_params, dropout_rng=None, train=False, output_hidden_states=True, output_attentions=None):
    inputs_embeds, prefix_len = _build_inputs_embeds(model, params, input_ids, local_patch, encoder_module, encoder_params)
    batch_size, total_len, _ = inputs_embeds.shape
    text_len = input_ids.shape[1]
    if prefix_len > 0:
        prefix_mask = jnp.ones((batch_size, prefix_len), dtype=attention_mask.dtype)
        attention_mask = jnp.concatenate([prefix_mask, attention_mask], axis=1)
    position_ids = jnp.broadcast_to(jnp.arange(total_len)[None, :], (batch_size, total_len))
    rngs = {"dropout": dropout_rng} if (dropout_rng is not None and train) else None
    model_output = model.module.apply(
        {"params": params},
        inputs_embeds,
        attention_mask,
        position_ids,
        not train,
        bool(output_attentions),
        bool(output_hidden_states),
        rngs=rngs,
        method=_lm_forward_with_embeds,
    )
    # Crop to text-aligned positions so downstream token-logprob code still works.
    cropped_logits = model_output.logits[:, prefix_len:prefix_len + text_len, :]
    hidden_states = None
    if model_output.hidden_states is not None:
        hidden_states = tuple(h[:, prefix_len:prefix_len + text_len, :] for h in model_output.hidden_states)
    model_output = model_output.replace(logits=cropped_logits, hidden_states=hidden_states)
    return model_output


class GPT2PPOTrain(PPOTrain):
    @classmethod
    def load_train(
        cls,
        policy_train_state: TrainState,
        value_head_train_state: TrainState,
        policy_model: FlaxPreTrainedModel,
        value_head_model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        loss_fn: Callable,
        bc_loss_fn: Optional[Callable] = None,
        bc_loss_weight: float = 0.0,
        patch_size: int = 3,
        num_visual_tokens: int = 4,
    ) -> "GPT2PPOTrain":
        mesh = policy_model.config.mesh
        assert mesh is not None
        assert mesh == value_head_model.config.mesh
        policy_pspec = match_partition_rules(policy_model.config.get_partition_rules(), policy_train_state)
        value_pspec = match_partition_rules(value_head_model.config.get_partition_rules(), value_head_train_state)
        encoder_module = LocalPatchCNN(num_cell_types=3, num_visual_tokens=num_visual_tokens, gpt_hidden_size=policy_model.config.n_embd)
        encoder_params = encoder_module.init(jax.random.PRNGKey(0), jnp.zeros((1, patch_size, patch_size), dtype=jnp.int32), train=False)

        @partial(
            pjit,
            donate_argnums=(0, 1),
            static_argnames=('train',),
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), policy_pspec),
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), value_pspec),
                NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()),
                NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()),
                NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()),
                NamedSharding(mesh, PS()), NamedSharding(mesh, PS()),
            ),
            out_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), policy_pspec),
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), value_pspec),
                NamedSharding(mesh, PS()), NamedSharding(mesh, PS()),
            ),
        )
        def _step(policy_train_state, value_head_train_state, input_ids, attention_mask, position_ids, local_patch,
                  should_take_action, old_logprobs, old_values, old_advantages, old_returns, prng_key,
                  bc_data_input_ids, bc_data_input_attention_mask, bc_data_input_position_ids, bc_data_input_training_mask,
                  train: bool = True):
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(('dp', 'fsdp'), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(('dp', 'fsdp'), None))
            if local_patch is not None:
                local_patch = with_named_sharding_constraint(local_patch, mesh, PS(('dp', 'fsdp'), None, None))
            should_take_action = with_named_sharding_constraint(should_take_action, mesh, PS(('dp', 'fsdp'), None))
            old_logprobs = with_named_sharding_constraint(old_logprobs, mesh, PS(('dp', 'fsdp'), None))
            old_values = with_named_sharding_constraint(old_values, mesh, PS(('dp', 'fsdp'), None))
            old_advantages = with_named_sharding_constraint(old_advantages, mesh, PS(('dp', 'fsdp'), None))
            old_returns = with_named_sharding_constraint(old_returns, mesh, PS(('dp', 'fsdp'), None))

            def grad_loss(policy_params: PyTree, value_head_params: PyTree, prng_key):
                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                model_output = _run_model(policy_model, policy_params, input_ids, attention_mask, local_patch, encoder_module, encoder_params, new_key, train, True, None)
                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                values = value_head_model.apply({'params': value_head_params}, model_output.hidden_states[-1], train=train, rngs={'dropout': new_key} if new_key is not None else None)[:, :-1]
                values = jnp.squeeze(values, axis=-1)
                logits = model_output.logits.astype(jnp.float32)
                logprobs = -softmax_cross_entropy_with_integer_labels(logits[:, :-1], input_ids[:, 1:])
                return loss_fn(attention_mask[:, 1:], logprobs, values, should_take_action, old_logprobs, old_values, old_advantages, old_returns)

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            (loss, info), (policy_grads, value_grads) = jax.value_and_grad(grad_loss, has_aux=True, argnums=(0, 1))(policy_train_state.params, value_head_train_state.params, new_key)
            policy_grads = jax.tree_util.tree_map(lambda x, ps: with_named_sharding_constraint(x, mesh, ps), policy_grads, policy_pspec.params)
            value_grads = jax.tree_util.tree_map(lambda x, ps: with_named_sharding_constraint(x, mesh, ps), value_grads, value_pspec.params)
            policy_train_state = policy_train_state.apply_gradients(grads=policy_grads)
            value_head_train_state = value_head_train_state.apply_gradients(grads=value_grads)
            return policy_train_state, value_head_train_state, loss, info

        obj = cls(policy_train_state=policy_train_state, value_head_train_state=value_head_train_state, policy_model=policy_model, value_head_model=value_head_model, tokenizer=tokenizer, _step=_step)
        object.__setattr__(obj, 'patch_size', patch_size)
        return obj

    def step(self, input_ids, should_take_action, old_logprobs, old_values, old_advantages, old_returns, prng_key,
             local_patch=None, attention_mask=None, position_ids=None,
             bc_data_input_ids=None, bc_data_input_attention_mask=None, bc_data_input_position_ids=None, bc_data_input_training_mask=None,
             train: bool = True):
        attention_mask, position_ids = initialize_attn_mask_pos_ids(input_ids, self.tokenizer.pad_token_id, attention_mask, position_ids)
        if bc_data_input_ids is not None:
            bc_data_input_attention_mask, bc_data_input_position_ids = initialize_attn_mask_pos_ids(bc_data_input_ids, self.tokenizer.pad_token_id, bc_data_input_attention_mask, bc_data_input_position_ids)
        policy_train_state, value_head_train_state, loss, logs = self._step(self.policy_train_state, self.value_head_train_state, input_ids, attention_mask, position_ids, local_patch, should_take_action, old_logprobs, old_values, old_advantages, old_returns, prng_key, bc_data_input_ids, bc_data_input_attention_mask, bc_data_input_position_ids, bc_data_input_training_mask, train)
        return self.replace(policy_train_state=policy_train_state, value_head_train_state=value_head_train_state), loss, logs


class PPOForwardOutputGPT2(NamedTuple):
    initial_policy_raw_output: FlaxCausalLMOutputWithCrossAttentions
    policy_raw_output: FlaxCausalLMOutputWithCrossAttentions
    values: jax.Array


class GPT2PPOInference(PPOInference):
    @classmethod
    def load_inference(
        cls,
        initial_policy_params: Optional[PyTree],
        policy_params: PyTree,
        value_head_params: PyTree,
        initial_policy_model: Optional[FlaxPreTrainedModel],
        policy_model: FlaxPreTrainedModel,
        value_head_model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        loss_fn: Optional[Callable],
        dp_shard_logits: bool = True,
        bc_loss_fn: Optional[Callable] = None,
        bc_loss_weight: float = 0.0,
        patch_size: int = 3,
        num_visual_tokens: int = 4,
    ) -> "GPT2PPOInference":
        mesh = policy_model.config.mesh
        assert mesh is not None
        assert mesh == value_head_model.config.mesh
        has_initial_policy = initial_policy_params is not None and initial_policy_model is not None
        initial_pspec = match_partition_rules(initial_policy_model.config.get_partition_rules(), initial_policy_params) if has_initial_policy else None
        policy_pspec = match_partition_rules(policy_model.config.get_partition_rules(), policy_params)
        value_pspec = match_partition_rules(value_head_model.config.get_partition_rules(), value_head_params)
        encoder_module = LocalPatchCNN(num_cell_types=3, num_visual_tokens=num_visual_tokens, gpt_hidden_size=policy_model.config.n_embd)
        encoder_params = encoder_module.init(jax.random.PRNGKey(0), jnp.zeros((1, patch_size, patch_size), dtype=jnp.int32), train=False)

        @partial(
            pjit,
            static_argnames=('initial_policy_output_attentions', 'initial_policy_output_hidden_states', 'policy_output_attentions', 'train'),
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), initial_pspec) if has_initial_policy else NamedSharding(mesh, PS()),
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), policy_pspec),
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), value_pspec),
                NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()),
            ),
            out_shardings=PPOForwardOutputGPT2(
                initial_policy_raw_output=FlaxCausalLMOutputWithCrossAttentions(logits=NamedSharding(mesh, PS(("dp", "fsdp"), None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), past_key_values=NamedSharding(mesh, PS()), hidden_states=NamedSharding(mesh, PS()), attentions=NamedSharding(mesh, PS()), cross_attentions=NamedSharding(mesh, PS())) if has_initial_policy else NamedSharding(mesh, PS()),
                policy_raw_output=FlaxCausalLMOutputWithCrossAttentions(logits=NamedSharding(mesh, PS(("dp", "fsdp"), None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), past_key_values=NamedSharding(mesh, PS()), hidden_states=NamedSharding(mesh, PS()), attentions=NamedSharding(mesh, PS()), cross_attentions=NamedSharding(mesh, PS())),
                values=NamedSharding(mesh, PS()),
            ),
        )
        def _forward(initial_policy_params, policy_params, value_head_params, input_ids, attention_mask, position_ids, local_patch=None, prng_key=None, initial_policy_output_attentions=None, initial_policy_output_hidden_states=None, policy_output_attentions=None, train=False):
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(("dp", "fsdp"), None))
            if local_patch is not None:
                local_patch = with_named_sharding_constraint(local_patch, mesh, PS(("dp", "fsdp"), None, None))
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            initial_output = None
            if has_initial_policy:
                initial_output = _run_model(initial_policy_model, initial_policy_params, input_ids, attention_mask, local_patch, encoder_module, encoder_params, new_key, train, initial_policy_output_hidden_states, initial_policy_output_attentions)
                initial_output = initial_output.replace(logits=initial_output.logits.at[:, :, initial_policy_model.config.unpadded_vocab_size:].set(-float('inf')))
            model_output = _run_model(policy_model, policy_params, input_ids, attention_mask, local_patch, encoder_module, encoder_params, new_key, train, True, policy_output_attentions)
            model_output = model_output.replace(logits=model_output.logits.at[:, :, policy_model.config.unpadded_vocab_size:].set(-float('inf')))
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            values = value_head_model.apply({'params': value_head_params}, model_output.hidden_states[-1], train=train, rngs={'dropout': new_key} if new_key is not None else None)
            values = jnp.squeeze(values, axis=-1)
            return PPOForwardOutputGPT2(initial_policy_raw_output=initial_output, policy_raw_output=model_output, values=values)

        @partial(
            pjit,
            static_argnames=('train',),
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), policy_pspec),
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), value_pspec),
                NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()), NamedSharding(mesh, PS()),
            ),
            out_shardings=(NamedSharding(mesh, PS()), NamedSharding(mesh, PS())),
        )
        def _eval_loss(policy_params, value_head_params, input_ids, attention_mask, position_ids, local_patch, should_take_action, old_logprobs, old_values, old_advantages, old_returns, prng_key, bc_data_input_ids, bc_data_input_attention_mask, bc_data_input_position_ids, bc_data_input_training_mask, train=False):
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(("dp", "fsdp"), None))
            if local_patch is not None:
                local_patch = with_named_sharding_constraint(local_patch, mesh, PS(("dp", "fsdp"), None, None))
            should_take_action = with_named_sharding_constraint(should_take_action, mesh, PS(("dp", "fsdp"), None))
            old_logprobs = with_named_sharding_constraint(old_logprobs, mesh, PS(("dp", "fsdp"), None))
            old_values = with_named_sharding_constraint(old_values, mesh, PS(("dp", "fsdp"), None))
            old_advantages = with_named_sharding_constraint(old_advantages, mesh, PS(("dp", "fsdp"), None))
            old_returns = with_named_sharding_constraint(old_returns, mesh, PS(("dp", "fsdp"), None))
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            model_output = _run_model(policy_model, policy_params, input_ids, attention_mask, local_patch, encoder_module, encoder_params, new_key, train, True, None)
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            values = value_head_model.apply({'params': value_head_params}, model_output.hidden_states[-1], train=train, rngs={'dropout': new_key} if new_key is not None else None)[:, :-1]
            values = jnp.squeeze(values, axis=-1)
            logits = model_output.logits.astype(jnp.float32)
            logprobs = -softmax_cross_entropy_with_integer_labels(logits[:, :-1], input_ids[:, 1:])
            return loss_fn(attention_mask[:, 1:], logprobs, values, should_take_action, old_logprobs, old_values, old_advantages, old_returns)

        obj = cls(initial_policy_params=initial_policy_params, policy_params=policy_params, value_head_params=value_head_params, initial_policy_model=initial_policy_model, policy_model=policy_model, value_head_model=value_head_model, tokenizer=tokenizer, _forward=_forward, _eval_loss=_eval_loss)
        object.__setattr__(obj, 'patch_size', patch_size)
        object.__setattr__(obj, 'num_visual_tokens', num_visual_tokens)
        object.__setattr__(obj, 'encoder_module', encoder_module)
        object.__setattr__(obj, 'encoder_params', encoder_params)
        return obj

    def forward(self, input_ids, attention_mask=None, position_ids=None, local_patch=None, initial_policy_output_attentions=None, initial_policy_output_hidden_states=None, policy_output_attentions=None, train=False, prng_key=None):
        attention_mask, position_ids = initialize_attn_mask_pos_ids(input_ids, self.tokenizer.pad_token_id, attention_mask, position_ids)
        return self._forward(self.initial_policy_params, self.policy_params, self.value_head_params, input_ids, attention_mask, position_ids, local_patch, prng_key, initial_policy_output_attentions, initial_policy_output_hidden_states, policy_output_attentions, train)

    def eval_loss(self, input_ids, should_take_action, old_logprobs, old_values, old_advantages, old_returns, local_patch=None, attention_mask=None, position_ids=None, prng_key=None, bc_data_input_ids=None, bc_data_input_attention_mask=None, bc_data_input_position_ids=None, bc_data_input_training_mask=None, train=False):
        attention_mask, position_ids = initialize_attn_mask_pos_ids(input_ids, self.tokenizer.pad_token_id, attention_mask, position_ids)
        return self._eval_loss(self.policy_params, self.value_head_params, input_ids, attention_mask, position_ids, local_patch, should_take_action, old_logprobs, old_values, old_advantages, old_returns, prng_key, bc_data_input_ids, bc_data_input_attention_mask, bc_data_input_position_ids, bc_data_input_training_mask, train)

    def get_ppo_data_from_text_trajectory_chain(self, text_trajectory_chains: List[TextTrajectoryChain], bsize: int, max_length: Optional[int] = None, train: bool = False, prng_key=None, token_process=None, verbose: bool = True, *, gamma, lam, kl_weight, use_advantage_whitening: bool = True, use_new_advantage_whitening: bool = False):
        assert self.initial_policy_model is not None and self.initial_policy_params is not None
        if token_process is None:
            token_process = lambda x: x
        token_trajectory_chains = [TokenTrajectoryChain.from_text_trajectory_chain(item, self.tokenizer, token_process=token_process) for item in text_trajectory_chains]
        patch_chunks = _extract_chain_patches(text_trajectory_chains, self.patch_size)

        n_chains = len(token_trajectory_chains)
        tokens = []
        combined = []
        for chain in token_trajectory_chains:
            combined.append(CombinedTokenTrajectoryChain.from_token_trajectory_chain(chain, max_length=max_length-1 if max_length is not None else None))
            tokens.extend(list(map(lambda x: x.tokens, chain.to_list())))
        tokens = block_sequences(tokens, pad_value=self.tokenizer.pad_token_id, dtype=np.int32, blocking_strategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=max_length))
        local_patches = np.stack(patch_chunks, axis=0)

        initial_policy_logprobs, policy_logprobs, values = [], [], []
        for i in range(0, len(tokens), bsize):
            tokens_batch = jnp.asarray(tokens[i:i+bsize], dtype=jnp.int32)
            patches_batch = jnp.asarray(local_patches[i:i+bsize], dtype=jnp.int32)
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            fb = self.forward(tokens_batch, local_patch=patches_batch, train=train, prng_key=new_key)
            init_lp = self.token_logprobs_from_logits(fb.initial_policy_raw_output.logits, tokens_batch)
            pol_lp = self.token_logprobs_from_logits(fb.policy_raw_output.logits, tokens_batch)
            initial_policy_logprobs.append(np.asarray(multihost_device_get(init_lp, mesh=self.initial_policy_model.config.mesh)))
            policy_logprobs.append(np.asarray(multihost_device_get(pol_lp, mesh=self.policy_model.config.mesh)))
            values.append(np.asarray(fb.values))
        initial_policy_logprobs = np.concatenate(initial_policy_logprobs, axis=0)
        policy_logprobs = np.concatenate(policy_logprobs, axis=0)
        values = np.concatenate(values, axis=0)

        batch_sections = list(map(lambda x: len(x.chunk_lens), combined))
        mask_split = np.split((tokens != self.tokenizer.pad_token_id), np.cumsum(batch_sections)[:-1], axis=0)
        init_split = np.split(initial_policy_logprobs, np.cumsum(batch_sections)[:-1], axis=0)
        pol_split = np.split(policy_logprobs, np.cumsum(batch_sections)[:-1], axis=0)
        val_split = np.split(values, np.cumsum(batch_sections)[:-1], axis=0)

        def unpad_array(arr, mask):
            return arr[mask.astype(bool)]

        init_chains = [np.concatenate([unpad_array(x, m) for x, m in zip(item, mask[:, 1:])], axis=0) for mask, item in zip(mask_split, init_split)]
        pol_chains = [np.concatenate([unpad_array(x, m) for x, m in zip(item, mask[:, 1:])], axis=0) for mask, item in zip(mask_split, pol_split)]
        val_chains = [np.concatenate([unpad_array(x, m)[:-1] for x, m in zip(item, mask)], axis=0) for mask, item in zip(mask_split, val_split)]
        last_values = [unpad_array(item[-1], mask[-1])[-1] for mask, item in zip(mask_split, val_split)]
        val_chains = [np.concatenate((item, last_values[i][None] * (1.0 - float(combined[i].done))), axis=0) for i, item in enumerate(val_chains)]

        log_ratio = [(p - q) * chain.should_take_action.astype(np.float32) for q, p, chain in zip(init_chains, pol_chains, combined)]
        valid_idxs = np.argwhere(np.concatenate([chain.should_take_action.astype(np.float32).reshape(-1) for chain in combined], axis=0))[:, 0]
        all_log_ratio = np.concatenate([x.reshape(-1) for x in log_ratio], axis=0)[valid_idxs]
        all_kls = np.exp(all_log_ratio) - 1 - all_log_ratio
        for i in range(n_chains):
            combined[i] = combined[i]._replace(rewards=combined[i].rewards - kl_weight * log_ratio[i])

        all_advantages, all_returns = [], []
        for i in range(n_chains):
            action_idxs, state_idxs, next_state_idxs = get_action_state_next_state_idxs(combined[i].should_take_action)
            state_values = val_chains[i][state_idxs]
            next_state_values = val_chains[i][next_state_idxs]
            action_rewards = combined[i].rewards[action_idxs]
            adv, ret = get_advantages_and_returns(state_values=state_values[None], next_state_values=next_state_values[None], action_rewards=action_rewards[None], gamma=gamma, lam=lam, use_whitening=False)
            all_advantages.append(adv[0]); all_returns.append(ret[0])
        if use_advantage_whitening:
            whitened = whiten(np.concatenate(all_advantages, axis=0), shift_mean=True)
            curr = 0
            for i in range(n_chains):
                L = all_advantages[i].shape[0]
                all_advantages[i] = whitened[curr:curr+L]
                curr += L
        advantage_chains, return_chains = [], []
        for i in range(n_chains):
            action_idxs, _, _ = get_action_state_next_state_idxs(combined[i].should_take_action)
            adv_chain = np.zeros((val_chains[i].shape[0]-1,), dtype=np.float32)
            adv_chain[action_idxs] = all_advantages[i]
            ret_chain = np.zeros((val_chains[i].shape[0]-1,), dtype=np.float32)
            ret_chain[action_idxs] = all_returns[i]
            advantage_chains.append(adv_chain); return_chains.append(ret_chain)

        ppo_datas = []
        patch_idx = 0
        for i in range(n_chains):
            input_ids_chunks = [x.tokens[:max_length] for x in token_trajectory_chains[i].to_list()]
            should_take_action_chunks = combined[i].unroll_arr(combined[i].should_take_action)
            old_logprobs_chunks = combined[i].unroll_arr(pol_chains[i])
            old_values_chunks = combined[i].unroll_arr(val_chains[i][:-1])
            old_advantages_chunks = combined[i].unroll_arr(advantage_chains[i])
            old_returns_chunks = combined[i].unroll_arr(return_chains[i])
            for chunk_idx in range(len(combined[i].chunk_lens)):
                ppo_datas.append(PPOData(
                    input_ids=input_ids_chunks[chunk_idx],
                    should_take_action=should_take_action_chunks[chunk_idx],
                    old_logprobs=old_logprobs_chunks[chunk_idx],
                    old_values=old_values_chunks[chunk_idx],
                    old_advantages=old_advantages_chunks[chunk_idx],
                    old_returns=old_returns_chunks[chunk_idx],
                    local_patch=patch_chunks[patch_idx],
                ))
                patch_idx += 1
        return ppo_datas, all_kls


class GPT2PPOPolicyMultimodal(PPOPolicy):
    def __init__(self, inference: GPT2PPOInference, prng_key, generation_config: Optional[GenerationConfig] = None, blocking_strategy: BlockingStrategy = BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), in_str_process=None, out_str_process=None, input_token_process=None, target_token_process=None, trace: bool = True):
        self.inference = inference
        self.prng_key = prng_key
        self.generation_config = generation_config
        self.blocking_strategy = blocking_strategy
        self.in_str_process = in_str_process or (lambda x: x)
        self.out_str_process = out_str_process or (lambda x: x)
        self.input_token_process = input_token_process
        self.target_token_process = target_token_process
        self.trace = trace

    def act(self, text_history: List[Optional[TextHistory]], done: Optional[List[bool]] = None):
        if done is None:
            done = [False] * len(text_history)
        eos_token = self.inference.tokenizer.eos_token or self.inference.tokenizer.pad_token or ''
        if self.generation_config is not None and self.generation_config.eos_token_id is not None:
            eos_token = self.inference.tokenizer.decode(self.generation_config.eos_token_id)
        raw_input_strs = [eos_token if d else self.in_str_process(text_history_to_str(item)) for item, d in zip(text_history, done)]
        new_key = None
        if self.prng_key is not None:
            self.prng_key, new_key = jax.random.split(self.prng_key)
        if getattr(self, '_text_inference', None) is None:
            from JaxSeq.models.gpt2.interface import GPT2Inference
            self._text_inference = GPT2Inference.load_inference(params=self.inference.policy_params, model=self.inference.policy_model, tokenizer=self.inference.tokenizer)
        model_outputs = self._text_inference.generate_from_str(input_strs=raw_input_strs, prng_key=new_key, blocking_strategy=self.blocking_strategy, generation_config=self.generation_config, input_token_process=self.input_token_process, target_token_process=self.target_token_process, trace=self.trace)
        raw_output_strs = model_outputs.output_strs
        output_strs = ["" if d else self.out_str_process(strip_prompt_from_completion(inp, out)) for inp, out, d in zip(raw_input_strs, raw_output_strs, done)]
        return [None if d else item + (Text(out, True),) for item, out, d in zip(text_history, output_strs, done)]

    def set_params(self, policy_params: PyTree) -> None:
        self.inference = self.inference.replace(policy_params=policy_params)
        if getattr(self, '_text_inference', None) is not None:
            self._text_inference = self._text_inference.replace(params=policy_params)
