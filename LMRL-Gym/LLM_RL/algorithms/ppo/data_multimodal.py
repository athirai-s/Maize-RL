from __future__ import annotations
from typing import Dict, Iterable, List, Iterator, NamedTuple, Optional
from JaxSeq.utils import Dataset, IterableDataset, block_sequences, BlockingStrategy
import numpy as np
import jax.numpy as jnp
import jax
from transformers.tokenization_utils import PreTrainedTokenizerBase


class PPOData(NamedTuple):
    input_ids: np.ndarray  # [t]
    should_take_action: np.ndarray  # [t-1]
    old_logprobs: np.ndarray  # [t-1]
    old_values: np.ndarray  # [t-1]
    old_advantages: np.ndarray  # [t-1]
    old_returns: np.ndarray  # [t-1]
    local_patch: Optional[np.ndarray] = None  # [H, W]

    @staticmethod
    def block(
        data: List["PPOData"],
        blocking_strategy: BlockingStrategy,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Dict[str, np.ndarray]:
        result = dict(
            input_ids=block_sequences(
                [x.input_ids for x in data],
                tokenizer.pad_token_id,
                dtype=np.int32,
                blocking_strategy=blocking_strategy,
            ),
            should_take_action=block_sequences(
                [x.should_take_action for x in data],
                False,
                dtype=np.bool_,
                blocking_strategy=blocking_strategy._replace(max_length=blocking_strategy.max_length - 1),
            ),
            old_logprobs=block_sequences(
                [x.old_logprobs for x in data],
                0.0,
                dtype=np.float32,
                blocking_strategy=blocking_strategy._replace(max_length=blocking_strategy.max_length - 1),
            ),
            old_values=block_sequences(
                [x.old_values for x in data],
                0.0,
                dtype=np.float32,
                blocking_strategy=blocking_strategy._replace(max_length=blocking_strategy.max_length - 1),
            ),
            old_advantages=block_sequences(
                [x.old_advantages for x in data],
                0.0,
                dtype=np.float32,
                blocking_strategy=blocking_strategy._replace(max_length=blocking_strategy.max_length - 1),
            ),
            old_returns=block_sequences(
                [x.old_returns for x in data],
                0.0,
                dtype=np.float32,
                blocking_strategy=blocking_strategy._replace(max_length=blocking_strategy.max_length - 1),
            ),
        )
        if any(x.local_patch is not None for x in data):
            patches = []
            fallback_shape = None
            for x in data:
                if x.local_patch is not None:
                    fallback_shape = x.local_patch.shape
                    break
            for x in data:
                if x.local_patch is None:
                    patches.append(np.ones(fallback_shape, dtype=np.int32))
                else:
                    patches.append(np.asarray(x.local_patch, dtype=np.int32))
            result['local_patch'] = np.stack(patches, axis=0)
        return result


class PPODataset(Dataset):
    def __init__(
        self,
        input_ids: np.ndarray,
        should_take_action: np.ndarray,
        old_logprobs: np.ndarray,
        old_values: np.ndarray,
        old_advantages: np.ndarray,
        old_returns: np.ndarray,
        local_patch: Optional[np.ndarray] = None,
    ):
        assert input_ids.shape[1] == (should_take_action.shape[1] + 1)
        assert input_ids.shape[1] == (old_logprobs.shape[1] + 1)
        assert input_ids.shape[1] == (old_values.shape[1] + 1)
        assert input_ids.shape[1] == (old_advantages.shape[1] + 1)
        assert input_ids.shape[1] == (old_returns.shape[1] + 1)
        assert input_ids.shape[0] == should_take_action.shape[0] == old_logprobs.shape[0] == old_values.shape[0] == old_advantages.shape[0] == old_returns.shape[0]
        if local_patch is not None:
            assert input_ids.shape[0] == local_patch.shape[0]

        self.input_ids = input_ids
        self.should_take_action = should_take_action
        self.old_logprobs = old_logprobs
        self.old_values = old_values
        self.old_advantages = old_advantages
        self.old_returns = old_returns
        self.local_patch = local_patch

    def __getitem__(self, index):
        item = {
            'input_ids': jnp.asarray(self.input_ids[index], dtype=jnp.int32),
            'should_take_action': jnp.asarray(self.should_take_action[index], dtype=jnp.bool_),
            'old_logprobs': jnp.asarray(self.old_logprobs[index], dtype=jnp.float32),
            'old_values': jnp.asarray(self.old_values[index], dtype=jnp.float32),
            'old_advantages': jnp.asarray(self.old_advantages[index], dtype=jnp.float32),
            'old_returns': jnp.asarray(self.old_returns[index], dtype=jnp.float32),
        }
        if self.local_patch is not None:
            item['local_patch'] = jnp.asarray(self.local_patch[index], dtype=jnp.int32)
        return item

    def __len__(self):
        return self.input_ids.shape[0]

    @classmethod
    def from_ppo_data_list(
        cls,
        ppo_data_list: List[PPOData],
        tokenizer: PreTrainedTokenizerBase,
        blocking_strategy: BlockingStrategy,
    ) -> "PPODataset":
        data = PPOData.block(ppo_data_list, blocking_strategy, tokenizer)
        return cls(**data)


class _PPOIteratorDataset:
    def __init__(self, ppo_data: Iterator[Dict[str, np.ndarray]]):
        self.ppo_data = ppo_data

    def __next__(self):
        item = next(self.ppo_data)
        out = {
            'input_ids': jnp.asarray(item['input_ids'], dtype=jnp.int32),
            'should_take_action': jnp.asarray(item['should_take_action'], dtype=jnp.bool_),
            'old_logprobs': jnp.asarray(item['old_logprobs'], dtype=jnp.float32),
            'old_values': jnp.asarray(item['old_values'], dtype=jnp.float32),
            'old_advantages': jnp.asarray(item['old_advantages'], dtype=jnp.float32),
            'old_returns': jnp.asarray(item['old_returns'], dtype=jnp.float32),
        }
        if 'local_patch' in item:
            out['local_patch'] = jnp.asarray(item['local_patch'], dtype=jnp.int32)
        return out


class PPOIterableDataset(IterableDataset):
    def __init__(self, ppo_data: Iterable[Dict[str, np.ndarray]]):
        self.ppo_data = ppo_data

    def __iter__(self):
        return _PPOIteratorDataset(iter(self.ppo_data))

    @classmethod
    def from_ppo_data_iterable(
        cls,
        ppo_data: Iterable[PPOData],
        tokenizer: PreTrainedTokenizerBase,
        blocking_strategy: BlockingStrategy,
    ) -> "PPOIterableDataset":
        class _TokensIterable(Iterable):
            def _tokens_generator(self):
                for item in ppo_data:
                    blocked = PPOData.block([item], blocking_strategy, tokenizer)
                    yield jax.tree_util.tree_map(lambda x: x[0], blocked)

            def __iter__(self):
                return self._tokens_generator()

        return cls(_TokensIterable())
