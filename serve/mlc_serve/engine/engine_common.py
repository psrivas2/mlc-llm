"""
Common utilites for engine classes.
"""

import time
from typing import Tuple, Deque, Dict, Optional, Union
from collections import deque
from threading import Condition, Lock

import structlog

from .base import (
    Request,
    RequestId,
    RequestState,
    GenerationSequence,
    SequenceId,
    StoppingCriteria,
    check_stopping_sequences,
)
from .metrics import PrometheusMetrics
from .metrics_labels import *
from .model_module import (
    DecodeRequest,
    PrefillRequest,
    Tokenizer,
    ConversationTemplate,
    KVCacheManager,
    ModelModule,
    TextGenerator,
    Tokenizer as TokenizerP,
)
from ..model.base import ModelArtifactConfig

LOG = structlog.stdlib.get_logger(__name__)


def get_new_request_state(
    request: Request, conversation_template: ConversationTemplate, tokenizer: Tokenizer
) -> RequestState:
    if request.debug_options.prompt is not None:
        prompt = request.debug_options.prompt
    else:
        prompt = conversation_template.apply(request.messages)

    prompt_tokens = tokenizer.encode(prompt)

    validation_err = None
    if request.validate_tokens is not None:
        validation_err = request.validate_tokens(request, prompt_tokens)

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
        validation_err=validation_err,
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


def update_sequence(
    gen_seq: GenerationSequence,
    new_token_ids: list[int],
    prompt_token_ids: list[int],
    tokenizer: Tokenizer,
    stopping_criteria: StoppingCriteria,
) -> str:
    gen_seq.next_start_position = len(prompt_token_ids + gen_seq.generated_token_ids)
    gen_seq.generated_token_ids.extend(new_token_ids)
    delta = decode_last_output(prompt_token_ids, gen_seq, tokenizer)
    gen_seq.output_text += delta

    gen_seq.output_text, delta, gen_seq.is_finished = check_stopping_sequences(
        stopping_criteria, gen_seq.output_text, delta, gen_seq.is_finished
    )

    return delta


def get_requests_to_process(
    current_states: list[RequestState], cache_manager: KVCacheManager
) -> Tuple[Union[list[PrefillRequest], list[DecodeRequest]], bool]:
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
        LOG.debug(
            "Creating prompt batch.",
            num_requests=len(requests),
            total_tokens=sum(len(r.token_ids) for r in requests),
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
        LOG.debug("Creating decode batch with %s requests.", len(requests))

    return requests, is_prompt_batch


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


class EngineBase:
    text_generator: TextGenerator
    tokenizer: TokenizerP
    model_artifact_config: ModelArtifactConfig
    max_context_length: int
    max_num_batched_tokens: int
    max_decode_steps: int
    min_decode_steps: int
    queue_lock: Lock
    queue: Deque[RequestState]
    has_new_requests: Condition
    current_batch: Dict[RequestId, RequestState]

    def __init__(self, model_module: ModelModule):
        self.text_generator = model_module.text_generator
        self.tokenizer = model_module.tokenizer
        self.conversation_template = model_module.conversation_template
        self.cache_manager = model_module.cache_manager
        self.model_artifact_config = model_module.model_artifact_config
        assert (
            self.model_artifact_config.max_context_length
        ), "max_context_length must not be zero"
        self.max_context_length = self.model_artifact_config.max_context_length
        self.max_num_batched_tokens = model_module.engine_config.max_num_batched_tokens
        self.max_decode_steps = min(
            self.cache_manager.get_kv_cache_size(),
            model_module.engine_config.max_decode_steps,
        )
        self.min_decode_steps = min(
            self.max_decode_steps - 1, model_module.engine_config.min_decode_steps
        )

        self.queue_lock = Lock()
        self.queue = deque[RequestState]()
        self.has_new_requests = Condition(lock=self.queue_lock)

        self.current_batch = dict[RequestId, RequestState]()

        self.prom_metrics = PrometheusMetrics()
        self.inv_kv_cache_size = 1.0 / self.cache_manager.get_kv_cache_size()

    def check_prompt_too_long(self, prompt_len: int, num_sequences: int = 1) -> bool:
        kv_cache_size = self.cache_manager.get_kv_cache_size()
        max_prompt_len = min(self.max_context_length, self.max_num_batched_tokens)

        # We make sure that the KV cache will have enough free space for this request to proceed
        # decoding for at least self.max_decode_steps steps.
        return (
            prompt_len > max_prompt_len
            or (kv_cache_size - prompt_len) < self.max_decode_steps * num_sequences
        )

    def evict_request(self):
        while self.cache_manager.get_max_new_tokens() < 1:
            self.prom_metrics.counter(NUM_CACHE_EVICTONS).inc()
            request_to_remove = min(
                self.current_batch.values(), key=lambda s: s.num_total_tokens
            )
            # TODO parallel sampling: Properly support Evicting a multi-sequence request
            assert (
                self.current_batch[request_to_remove.request_id].num_sequences == 1
            ), "Evicting a multi-sequence request is not supported."

            self.remove_request_from_batch(request_to_remove.request_id)
            self.queue.appendleft(request_to_remove)

            LOG.debug(
                "Preempt request to free %s tokens",
                request_to_remove.num_total_tokens,
            )

    def try_grow_batch(self, num_new_batched_tokens) -> Optional[int]:
        max_new_tokens = self.cache_manager.get_max_new_tokens()
        if max_new_tokens < self.min_decode_steps:
            LOG.debug(
                "Stop growing the batch due to min_decode_steps. Decode steps: %s",
                max_new_tokens,
            )
            # stop adding request if there isn't enough space to do a certain steps of decoding.
            return None

        state = self.queue[0]

        if state.num_sequences == 1:
            gen_seq = state.generation_sequences[0]
            num_tokens = state.prompt_len + len(gen_seq.generated_token_ids)
            num_new_batched_tokens += num_tokens
            # This can happen when we are recovering from cache eviction and the sum of prompt
            # and intermediate decode tokens is bigger than the biggest allowable batch size,
            # self.max_num_batched_tokens. In such cases, we need to discard the recent decode
            # tokens that cannot fit into a batch, and recompute them after we fill the cache
            # entries for the older tokens.
            if (
                len(self.current_batch) == 0
                and num_tokens > self.max_num_batched_tokens
            ):
                gen_seq.generated_token_ids = gen_seq.generated_token_ids[
                    : (self.max_num_batched_tokens - state.prompt_len)
                ]
                gen_seq.next_start_position = (
                    num_new_batched_tokens
                ) = num_tokens = self.max_num_batched_tokens
        else:
            # Evicting and recovering multi-sequence requests is not supported for now.
            assert all(
                gen_seq.next_start_position == 0
                for gen_seq in state.generation_sequences
            )
            num_tokens = state.prompt_len
            num_new_batched_tokens += num_tokens

        if num_new_batched_tokens > self.max_num_batched_tokens > 0:
            LOG.debug(
                "Stop growing the batch due to max_num_batched_tokens. Batched tokens: %s",
                num_new_batched_tokens,
            )
            return None

        # We make sure that the KV cache will have enough free space for this request to proceed
        # decoding for at least self.max_decode_steps steps.
        if (self.cache_manager.get_free_space() - num_tokens) / (
            len(self.current_batch) + 1
        ) < self.max_decode_steps * state.num_sequences:
            LOG.debug(
                "Stop growing the batch due to not enough free space. Free: %s, Num tokens: %s",
                self.cache_manager.get_free_space(),
                num_tokens,
            )
            return None

        self.queue.popleft()
        # TODO parallel sampling: Need update here when evicting multi-sequence requests is supported.
        self.cache_manager.allocate(state.request_id, num_tokens)
        self.current_batch[state.request_id] = state

        return num_new_batched_tokens

    def remove_request_from_batch(self, request_id: RequestId):
        self.cache_manager.free_request(self.current_batch[request_id])
        del self.current_batch[request_id]
