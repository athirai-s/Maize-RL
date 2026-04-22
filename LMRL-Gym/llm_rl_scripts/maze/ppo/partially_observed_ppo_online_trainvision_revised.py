from typing import Optional
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from transformers import AutoTokenizer
from JaxSeq.utils import convert_path, load_mesh, get_dtype, setup_experiment_save
import jax
import jax.numpy as jnp
from JaxSeq.utils import BlockingStrategy, Padding, Truncation, get_weight_decay_mask, create_path, get_enabled_save_path
import os
import optax
from JaxSeq.models.gpt2.interface import GPT2Inference
from JaxSeq.models.gpt2.load import load_train_state, ModelLoadMode
import pickle as pkl
from LLM_RL.algorithms.ppo.score_fn import build_ppo_score_fn
from LLM_RL.algorithms.ppo.train import train_loop
from LLM_RL.algorithms.ppo.base_interface import ppo_loss_fn, FixedKLController, AdaptiveKLController
from transformers.generation import GenerationConfig
from jaxtyping import PyTree
import re
from LLM_RL.environment import Text, TokenHistory, text_env_eval, TextTrajectory, TextTrajectoryChain
from LLM_RL.algorithms.ppo.gpt2_multimodal.interface import GPT2PPOInference, GPT2PPOTrain, GPT2PPOPolicyMultimodal
from LLM_RL.heads.linear_head import load_train_state_from_config as load_head_train_state_from_config
from LLM_RL.heads.linear_head import LinearHeadConfig
from JaxSeq.shard_model import shard_params_from_params
from LLM_RL.algorithms.ppo.data import PPODataset
from LLM_RL.utils import get_tensor_stats_np
from functools import partial
import numpy as np
from JaxSeq.logs import label_logs, log, pull_logs
import json
from JaxSeq.utils import multihost_device_get
from IPython import embed
from JaxSeq.data import MaskDataset
from JaxSeq.models.gpt2.interface import loss_fn_mask
from llm_rl_scripts.maze.env.env import maze_proposal_function
from LLM_RL.algorithms.ppo.reranker_policy import ReRankerSamplePolicy
from JaxSeq.utils import block_sequences
from llm_rl_scripts.maze.env.maze_utils import setup_maze_env, pick_start_position

# NEW: VISUAL
from llm_rl_scripts.maze.models.local_patch_encoder import LocalPatchCNN  # noqa: F401


def _build_generation_config(
    tokenizer,
    policy_do_sample: bool,
    policy_num_beams: int,
    policy_temperature: Optional[float],
    policy_top_p: Optional[float],
    policy_top_k: Optional[int],
    max_output_length: int,
) -> GenerationConfig:
    """Shared generation config for both text-only and multimodal policies."""
    return GenerationConfig(
        do_sample=policy_do_sample,
        num_beams=policy_num_beams,
        temperature=policy_temperature,
        top_p=policy_top_p,
        top_k=policy_top_k,
        eos_token_id=tokenizer.encode('\n')[0],
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_output_length,
    )


def main(
    model_load_mode: ModelLoadMode,
    model_load_path: str,

    # Mark the end of positional arguments.
    bc_data_path: Optional[str] = None,
    train_bc_bsize: int = 8,
    bc_loss_weight: int = 0,
    model_str: str = "gpt2",

    exp_name: Optional[str] = None,
    outputs_path: Optional[str] = None,
    maze_name: str = "medium",

    data_mesh_shape: int = 1,
    fsdp_mesh_shape: int = 1,
    model_mesh_shape: int = -1,

    use_wandb: bool = False,
    wandb_project: Optional[str] = "llm_rl_ppo_maze",

    n_rounds: int = 1,
    epochs: int = 1,
    max_steps: Optional[int] = None,

    lr: float = 1e-5,
    weight_decay: float = 0.0,

    train_bsize: int = 32,
    grad_accum_steps: int = 1,
    rollout_bsize: int = 32,
    n_rollouts: int = 128,
    ppo_data_bsize: int = 32,
    num_pos_per_setup: int = 4,

    gradient_checkpointing: bool = False,
    gradient_checkpointing_policy: str = 'nothing_saveable',
    use_fp16_activations: bool = False,
    use_fp16_params: bool = False,

    max_input_length: int = 512,
    max_output_length: int = 10,

    log_every: int = 256,
    eval_every_steps: Optional[int] = None,
    eval_every_epochs: Optional[int] = None,
    eval_every_rounds: Optional[int] = 1,
    eval_at_beginning: bool = True,
    eval_at_end: bool = True,

    save_every_steps: Optional[int] = None,
    save_every_epochs: Optional[int] = None,
    save_every_rounds: Optional[int] = 10,
    save_at_beginning: bool = False,
    save_at_end: bool = True,
    save_best: bool = True,
    max_checkpoints: Optional[int] = 20,
    save_train_state: bool = True,
    save_ppo_dataset: bool = True,
    save_bf16: bool = True,

    policy_do_sample: bool = True,
    policy_num_beams: int = 1,
    policy_temperature: Optional[float] = None,
    policy_top_p: Optional[float] = None,
    policy_top_k: Optional[int] = None,

    gamma: float = 1.0,
    lam: float = 0.95,
    use_advantage_whitening: bool = True,

    init_kl_coef: float = 0.001,
    kl_target: Optional[float] = None,
    kl_horizon: Optional[int] = None,

    cliprange_value: float = 0.2,
    cliprange: float = 0.2,
    value_loss_coef: float = 1.0,

    force_pad_embeddings: bool = False,

    should_restore_loop_state: bool = False,

    # IMPORTANT: this should stay partial-observation by default.
    # The repo often uses the generic string in examples, but your multimodal
    # setup is meant to extend the partially observed setting.
    describe_function: str = "describe_observation_only_walls",
    reranker_policy: bool = False,

    reward_function: str = "standard_reward",

    # NEW: VISUAL
    use_visual_patch: bool = True,
    patch_size: int = 3,
    print_local_patch: bool = True,
    num_visual_tokens: int = 4,
):
    input_args = locals().copy()
    print(input_args)

    use_adaptive_kl = (kl_target is not None and kl_horizon is not None)
    if not use_adaptive_kl:
        assert kl_target is None and kl_horizon is None

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    if bc_data_path is not None:
        with open(bc_data_path, 'rb') as f:
            text_histories = pkl.load(f)

        blocking_strategy = BlockingStrategy(Padding.RIGHT, Truncation.RIGHT, max_input_length + max_output_length)

        token_histories = list(map(lambda x: TokenHistory.from_text_history(x, tokenizer), text_histories))
        in_tokens = list(map(lambda x: block_sequences([x.tokens], tokenizer.pad_token_id, dtype=np.int32, blocking_strategy=blocking_strategy)[0], token_histories))
        is_actions = list(map(lambda x: block_sequences([x.is_action], 0.0, dtype=np.float32, blocking_strategy=blocking_strategy)[0], token_histories))

        bc_data = MaskDataset(
            in_tokens=jnp.array(in_tokens),
            in_training_mask=jnp.array(is_actions),
        )
    else:
        bc_data = None

    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")

    def policy_optim_getter(params: PyTree):
        mask = get_weight_decay_mask((
            "".join([r"\['ln_[0-9]+'\]", re.escape("['bias']")]),
            "".join([r"\['ln_[0-9]+'\]", re.escape("['scale']")]),
            re.escape("['ln_f']['bias']"),
            re.escape("['ln_f']['scale']"),
            "bias",
        ))(params)
        return optax.MultiSteps(
            optax.adamw(
                learning_rate=lr,
                b1=0.9,
                b2=0.95,
                eps=1e-8,
                weight_decay=weight_decay,
                mask=mask,
            ),
            every_k_schedule=grad_accum_steps,
        )

    model_dtype = get_dtype(use_fp16=use_fp16_activations)
    params_dtype = get_dtype(use_fp16=use_fp16_params)

    model_prng_key = jax.random.PRNGKey(2)
    policy_train_state, policy_model = load_train_state(
        model_load_mode=model_load_mode,
        model_load_path=convert_path(model_load_path) if model_load_mode != ModelLoadMode.HF else model_load_path,
        model_dtype=model_dtype,
        optim_getter=policy_optim_getter,
        tokenizer=tokenizer,
        mesh=mesh,
        prng_key=model_prng_key,
        force_pad_embeddings=force_pad_embeddings,
        params_dtype=params_dtype,
    )
    policy_model.config.gradient_checkpointing = gradient_checkpointing
    policy_model.config.gradient_checkpointing_policy = gradient_checkpointing_policy
    with jax.default_device(jax.devices('cpu')[0]):
        initial_policy_params = jax.tree_util.tree_map(
            lambda x: multihost_device_get(x, mesh=mesh).copy(),
            policy_train_state.params,
        )
    initial_policy_params = shard_params_from_params(
        model=policy_model,
        params=initial_policy_params,
    )

    loop_state = dict()
    if should_restore_loop_state and (model_load_mode in {ModelLoadMode.TRAIN_STATE,
                                                          ModelLoadMode.TRAIN_STATE_PARAMS,
                                                          ModelLoadMode.PARAMS}):
        with open(os.path.join(convert_path(model_load_path), 'loop_state.pkl'), 'rb') as f:
            loop_state = pkl.load(f)


    policy_prng = jax.random.PRNGKey(0)
    generation_config = _build_generation_config(
        tokenizer=tokenizer,
        policy_do_sample=policy_do_sample,
        policy_num_beams=policy_num_beams,
        policy_temperature=policy_temperature,
        policy_top_p=policy_top_p,
        policy_top_k=policy_top_k,
        max_output_length=max_output_length,
    )
    policy_blocking_strategy = BlockingStrategy(
        padding=Padding.LEFT,
        truncation=Truncation.LEFT,
        max_length=max_input_length,
    )

    # NEW: VISUAL
    # Training/evaluation rollouts must use the same multimodal policy wrapper
    if use_visual_patch:
        multimodal_policy_inference = GPT2PPOInferenceMultimodal.from_base_inference(
            base_inference=policy_inference,
            patch_size=patch_size,
            num_visual_tokens=num_visual_tokens,
        )
        policy = GPT2PPOPolicyMultimodal(
            inference=multimodal_policy_inference,
            prng_key=policy_prng,
            generation_config=generation_config,
            blocking_strategy=policy_blocking_strategy,
            out_str_process=lambda x: x.removesuffix('\n') + '\n',
        )
    else:
        policy = GPT2PPOPolicy(
            inference=policy_inference,
            prng_key=policy_prng,
            generation_config=generation_config,
            blocking_strategy=policy_blocking_strategy,
            out_str_process=lambda x: x.removesuffix('\n') + '\n',
        )

    def value_head_optim_getter(params: PyTree):
        mask = get_weight_decay_mask(("bias",))(params)
        return optax.MultiSteps(
            optax.adamw(
                learning_rate=lr,
                b1=0.9,
                b2=0.95,
                eps=1e-8,
                weight_decay=weight_decay,
                mask=mask,
            ),
            every_k_schedule=grad_accum_steps,
        )

    head_prng_key = jax.random.PRNGKey(3)
    value_head_train_state, value_head = load_head_train_state_from_config(
        model_config=LinearHeadConfig(
            input_dim=policy_model.config.n_embd,
            output_dim=1,
            use_bias=True,
            initializer_range=0.0,
            bias_init=-1,
        ),
        model_dtype=jnp.float32,
        optim_getter=value_head_optim_getter,
        mesh=mesh,
        prng_key=head_prng_key,
        pad_to_output_dim=None,
        params_dtype=jnp.float32,
    )

    loss_f = partial(ppo_loss_fn, cliprange_value=cliprange_value, cliprange=cliprange, value_loss_coef=value_loss_coef)

    ppo_inference = GPT2PPOInference.load_inference(
        initial_policy_params=initial_policy_params,
        policy_params=policy_train_state.params,
        value_head_params=value_head_train_state.params,
        initial_policy_model=policy_model,
        policy_model=policy_model,
        value_head_model=value_head,
        tokenizer=tokenizer,
        loss_fn=loss_f,
        bc_loss_fn=loss_fn_mask if bc_data is not None else None,
        bc_loss_weight=bc_loss_weight if bc_data is not None else 0.0,
        patch_size=patch_size,
        num_visual_tokens=num_visual_tokens,
    )

    ppo_trainer = GPT2PPOTrain.load_train(
        policy_train_state=policy_train_state,
        value_head_train_state=value_head_train_state,
        policy_model=policy_model,
        value_head_model=value_head,
        tokenizer=tokenizer,
        loss_fn=loss_f,
        bc_loss_fn=loss_fn_mask if bc_data is not None else None,
        bc_loss_weight=bc_loss_weight if bc_data is not None else 0.0,
        patch_size=patch_size,
        num_visual_tokens=num_visual_tokens,
    )

    policy = GPT2PPOPolicyMultimodal(
        inference=ppo_inference,
        prng_key=policy_prng,
        generation_config=generation_config,
        blocking_strategy=policy_blocking_strategy,
        out_str_process=lambda x: x.removesuffix('\n') + '\n',
    )

    if use_adaptive_kl:
        kl_controller = AdaptiveKLController(init_kl_coef=init_kl_coef, target=kl_target, horizon=kl_horizon)
    else:
        kl_controller = FixedKLController(kl_coef=init_kl_coef)

    maze_last_k = 40
    env = setup_maze_env(
        maze_name=maze_name,
        describe_function=describe_function,
        reward_function=reward_function,
        last_k=maze_last_k,
        patch_size=patch_size,            # NEW: VISUAL
        print_local_patch=print_local_patch,  # NEW: VISUAL
    )
    start_position = pick_start_position(maze_name=maze_name)

    data_round = 0

    def ppo_dataset_loader(ppo_inference: GPT2PPOInference, policy: GPT2PPOPolicy) -> PPODataset:
        if reranker_policy:
            print("reranker policy!")
            policy = ReRankerSamplePolicy(
                proposal_fn=maze_proposal_function,
                score_fn=build_ppo_score_fn(
                    inference=ppo_inference,
                    tokenizer=tokenizer,
                    max_length=max_input_length + max_output_length,
                    bsize=ppo_data_bsize,
                )
            )

        print("collecting data ...")
        nonlocal data_round
        raw_results, summary_results = text_env_eval(
            env=env,
            policy=policy,
            n_rollouts=n_rollouts,
            bsize=rollout_bsize,
            # env_options={"init_position": start_position},
        )
        summary_results = pull_logs(summary_results)

        text_trajectory_chains = []
        for raw_result in raw_results:
            curr_chain = []
            for transition in raw_result:
                try:
                    # NEW: VISUAL
                    # Preserve the latest observation extras (including local_patch)
                    # so rollout-time multimodal information is not silently discarded
                    # before building trajectory chains.
                    state_items = list(transition.post_action_history[:-1])
                    state_text = " ".join([item.text for item in state_items])
                    latest_obs_extras = None
                    for item in reversed(state_items):
                        if (not item.is_action) and (getattr(item, 'extras', None) is not None):
                            latest_obs_extras = item.extras
                            break
                    state = Text(state_text, False, extras=latest_obs_extras)
                    action = transition.post_action_history[-1]
                    text_trajectory = TextTrajectory(
                        text_history=[state, action],
                        reward=[0.0, transition.reward],
                        done=transition.done,
                    )
                    curr_chain.append(text_trajectory)
                except Exception:
                    embed()

            chain = None
            for text_trajectory in curr_chain[::-1]:
                chain = TextTrajectoryChain(
                    text_trajectory=text_trajectory,
                    next=chain,
                )

            text_trajectory_chains.append(chain)

        ppo_data, all_kls = ppo_inference.get_ppo_data_from_text_trajectory_chain(
            text_trajectory_chains,
            bsize=ppo_data_bsize,
            max_length=max_input_length + max_output_length,
            gamma=gamma,
            lam=lam,
            kl_weight=kl_controller.value,
            use_advantage_whitening=use_advantage_whitening,
        )
        mean_kl = all_kls.mean().item()
        kl_controller.update(mean_kl, train_bsize)

        ppo_dataset = PPODataset.from_ppo_data_list(
            ppo_data,
            tokenizer,
            BlockingStrategy(Padding.RIGHT, Truncation.RIGHT, max_input_length + max_output_length),
        )

        logs = dict(
            policy=dict(
                initial_policy_kl=get_tensor_stats_np(all_kls, np.ones(all_kls.shape), all_kls.size),
                sqrt_initial_policy_kl=np.sqrt(mean_kl),
                kl_ctrl_value=kl_controller.value,
            ),
            env_interaction=summary_results,
        )

        logs = pull_logs(label_logs(logs, 'data_collection', {'round': data_round}))
        log(logs, use_wandb and is_main_process)

        if save_dir is not None and save_ppo_dataset:
            print('saving ppo dataset ...')
            print(save_dir)
            data_save_path = os.path.join(save_dir, 'data_saves', f'{data_round}')
            if is_main_process:
                create_path(data_save_path)
            with open(get_enabled_save_path(
                os.path.join(data_save_path, 'ppo_dataset.pkl'),
                enabled=is_main_process,
            ), 'wb') as f:
                pkl.dump(ppo_dataset, f)
            with open(get_enabled_save_path(
                os.path.join(data_save_path, 'text_trajectory_chains.pkl'),
                enabled=is_main_process,
            ), 'wb') as f:
                pkl.dump(text_trajectory_chains, f)
            with open(get_enabled_save_path(
                os.path.join(data_save_path, 'raw_results.pkl'),
                enabled=is_main_process,
            ), 'wb') as f:
                pkl.dump(raw_results, f)
            with open(get_enabled_save_path(
                os.path.join(data_save_path, 'summary_results.json'),
                enabled=is_main_process,
            ), 'w') as f:
                json.dump(summary_results, f)
            print('done saving ppo dataset.')

        data_round += 1

        return ppo_dataset

    outputs_path = f"{outputs_path}/{exp_name}"
    save_dir, exp_name = setup_experiment_save(
        exp_name=exp_name,
        outputs_path=outputs_path,
        input_args=input_args,
        script__file__=__file__,
        is_main_process=is_main_process,
    )

    train_prng = jax.random.PRNGKey(1)
    save_dtype = jnp.bfloat16 if save_bf16 else jnp.float32
    ppo_trainer, ppo_inference, policy = train_loop(
        trainer=ppo_trainer,
        inference=ppo_inference,
        policy=policy,
        load_dataset=ppo_dataset_loader,
        evaluator=None,
        prng_key=train_prng,
        save_dir=save_dir,
        n_rounds=n_rounds,
        epochs=epochs,
        max_steps=max_steps,
        bsize=train_bsize,
        log_every=log_every,
        eval_every_steps=eval_every_steps,
        eval_every_epochs=eval_every_epochs,
        eval_every_rounds=eval_every_rounds,
        eval_at_beginning=eval_at_beginning,
        eval_at_end=eval_at_end,
        save_every_steps=save_every_steps,
        save_every_epochs=save_every_epochs,
        save_every_rounds=save_every_rounds,
        save_at_beginning=save_at_beginning,
        save_at_end=save_at_end,
        save_best=save_best,
        max_checkpoints=max_checkpoints,
        save_train_state=save_train_state,
        save_dtype=save_dtype,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=exp_name,
        wandb_config=None,
        is_main_process=is_main_process,
        bc_dataset=bc_data,
        bc_bsize=train_bc_bsize,
        **loop_state,
    )


if __name__ == "__main__":
    tyro.cli(main)
