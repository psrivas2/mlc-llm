"""
Common utilites for engine classes.
"""

from .base import (
    RequestState,
    GenerationSequence,
)
from .model_module import Tokenizer


def decode_last_output(
    prompt_tokens: list[int], generation_sequence: GenerationSequence, tokenizer: Tokenizer
) -> str:
    if len(generation_sequence.output_text):
        prefix_idx = max(0, generation_sequence.next_start_position - 6)
    else:
        prefix_idx = generation_sequence.next_start_position

    token_ids = prompt_tokens + generation_sequence.generated_token_ids

    if prefix_idx == 0:
        return tokenizer.decode(token_ids)

    prefix = tokenizer.decode(
        token_ids[prefix_idx : generation_sequence.next_start_position]
    )
    full = tokenizer.decode(token_ids[prefix_idx:])

    return full[len(prefix) :]


def should_stop_by_length(state: RequestState, max_context_length: int) -> bool:
    # TODO: currently, we simply return true for both stopping reasons.
    #       in the future, we can differentiate these two.
    # this include prompt tokens and gen tokens so far
    for gen_seq in state.generation_sequences:
        if gen_seq.is_finished:
            continue

        num_context_tokens = len(state.prompt_token_ids + gen_seq.generated_token_ids)
        if num_context_tokens >= max_context_length:
            gen_seq.is_finished = True
            continue

        num_gen_tokens = num_context_tokens - state.prompt_len
        if (
            state.stopping_criteria.max_tokens is not None
            and num_gen_tokens < state.stopping_criteria.max_tokens
        ):
            return False

    return True
