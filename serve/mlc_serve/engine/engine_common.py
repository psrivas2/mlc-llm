"""
Common utilites for engine classes.
"""

import time
import logging
from typing import Tuple

from .base import (
    Request,
    RequestState,
    GenerationSequence,
    SequenceId,
)
from .model_module import (
    DecodeRequest,
    PrefillRequest,
    Tokenizer,
    ConversationTemplate,
    KVCacheManager,
)


logger = logging.getLogger(__name__)


def get_requests_to_process(
    current_states: list[RequestState], cache_manager: KVCacheManager
) -> Tuple[list[Request], bool]:
    requests = []
    # TODO: consider having hybrid batch if the underlying attention kernel supports
    # mixing prefill and decode.
    is_prompt_batch = any(
        state.generation_sequences[0].next_start_position == 0
        for state in current_states
    )

    if is_prompt_batch:
        for state in current_states:
            if state.generation_sequences[0].next_start_position == 0:
                requests.append(
                    PrefillRequest(
                        request_id=state.request_id,
                        token_ids=state.prompt_token_ids,
                        num_sequence=state.num_sequences,
                        sampling_params=state.sampling_params,
                    )
                )
        logger.debug(
            "Creating prompt batch with %s requests with %s total tokens.",
            len(requests),
            sum(len(r.token_ids) for r in requests),
        )
    else:
        for state in current_states:
            for gen_seq in state.generation_sequences:
                if not gen_seq.is_finished:
                    token_ids = state.prompt_token_ids + gen_seq.generated_token_ids
                    requests.append(
                        DecodeRequest(
                            sequence_id=gen_seq.seq_id,
                            token_ids=token_ids,
                            sampling_params=state.sampling_params,
                        )
                    )
                    cache_manager.extend(
                        gen_seq.seq_id,
                        len(token_ids) - gen_seq.next_start_position,
                    )
        logger.debug("Creating decode batch with %s requests.", len(requests))

    return requests, is_prompt_batch


def get_new_request_state(
    request: Request, conversation_template: ConversationTemplate, tokenizer: Tokenizer
) -> RequestState:
    if request.debug_options.prompt is not None:
        prompt = request.debug_options.prompt
    else:
        prompt = conversation_template.apply(request.messages)

    prompt_tokens = tokenizer.encode(prompt)

    gen_seqs = [
        GenerationSequence(
            seq_id=SequenceId(request.request_id, i),
            generated_token_ids=[],
            next_start_position=0,
            output_text="",
        )
        for i in range(request.num_sequences)
    ]

    return RequestState(
        request_id=request.request_id,
        prompt_token_ids=prompt_tokens,
        generation_sequences=gen_seqs,
        sampling_params=request.sampling_params,
        stopping_criteria=request.stopping_criteria,
        debug_options=request.debug_options,
        arrival_timestamp=time.time(),
    )


def decode_last_output(
    prompt_tokens: list[int],
    generation_sequence: GenerationSequence,
    tokenizer: Tokenizer,
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
