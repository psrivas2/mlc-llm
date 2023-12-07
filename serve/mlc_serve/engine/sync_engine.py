"""
A implementation of InferenceEngine that executes in the current process.
"""

import time
import logging
from typing import Deque, Set, Dict
from collections import deque, defaultdict
from threading import Condition, Lock

from .base import (
    FinishReason,
    InferenceEngine,
    InferenceStepResult,
    Request,
    RequestId,
    RequestOutput,
    RequestState,
    SequenceOutput,
    check_stopping_sequences,
    ValidationError,
    GenerationSequence,
    SequenceId,
)
from .model_module import (
    DecodeRequest,
    ModelModule,
    PrefillRequest,
    TextGenerator,
    Tokenizer as TokenizerP,
)
from ..model.base import ModelArtifactConfig

logger = logging.getLogger(__name__)


class SynchronousInferenceEngine(InferenceEngine):
    """
    A implementation of InferenceEngine that does inference synchronously in the current thread
    when `step` is called.
    """

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
    requests_to_be_cancelled: Set[RequestId]
    current_batch: Dict[RequestId, RequestState]

    def __init__(
        self,
        model_module: ModelModule,
    ):
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
        assert (
            self.max_num_batched_tokens > 0
        ), "max_num_batched_tokens must be positive"
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
        self.requests_to_be_cancelled = set[RequestId]()

        self.current_batch = dict[RequestId, RequestState]()

    def add(self, requests: list[Request]):
        if not requests:
            return []

        new_request_states = []
        for req in requests:
            # TODO: verify that request id is unique
            # wrap the stop sequence with list if necessary
            if req.stopping_criteria.stop_sequences:
                if isinstance(req.stopping_criteria.stop_sequences, str):
                    req.stopping_criteria.stop_sequences = [
                        req.stopping_criteria.stop_sequences
                    ]
                assert isinstance(req.stopping_criteria.stop_sequences, list)

            state = self._get_new_request_state(req)
            new_request_states.append(state)

            if (
                state.validation_err is not None
                or state.prompt_len
                > min(self.max_context_length, self.max_num_batched_tokens)
                # We make sure that the KV cache will have enough free space for this request to proceed
                # decoding for at least self.max_decode_steps steps.
                or self.cache_manager.get_kv_cache_size() - state.prompt_len
                < self.max_decode_steps * req.num_sequences
            ):
                self.cancel(req.request_id)
                if state.validation_err is None:
                    state.validation_err = ValidationError(
                        "The prompt is too long for the given set of engine parameters."
                    )

        with self.queue_lock:
            self.queue.extend(new_request_states)
            self.has_new_requests.notify_all()

    def cancel(self, request_id: RequestId):
        with self.queue_lock:
            # TODO: consider iterating throught the queue to find if request id exist
            # Otherwise cancel a request that's already finished will leave request_id
            # in the `requests_to_be_cancelled` set forever.
            self.requests_to_be_cancelled.add(request_id)

    def wait_for_request(self, timeout_seconds=None) -> bool:
        with self.queue_lock:
            return self.has_new_requests.wait_for(
                self.has_pending_requests, timeout=timeout_seconds
            )

    def step(self) -> InferenceStepResult:
        logger.debug("Starting new inference step.")

        outputs = list[RequestOutput]()

        # TODO: consolidate into a single function
        for state in list(self.current_batch.values()):
            finish_reason = None
            if state.is_finished:
                finish_reason = FinishReason.Stop
            if self._should_stop_by_length(state):
                finish_reason = FinishReason.Length

            if finish_reason is not None:
                seq_outputs = [
                    SequenceOutput(
                        i,
                        finish_reason=finish_reason,
                        num_generated_tokens=len(gen_seq.generated_token_ids),
                    )
                    for i, gen_seq in enumerate(state.generation_sequences)
                ]

                outputs.append(
                    RequestOutput(
                        state.request_id,
                        seq_outputs,
                        num_prompt_tokens=state.prompt_len,
                    )
                )
                self.current_batch.pop(state.request_id)
                self.cache_manager.free_request(state)

        previous_requests_to_be_cancelled = set(self.requests_to_be_cancelled)
        self._adjust_batch()

        if not self.current_batch:
            if len(self.queue) > 0:
                logger.warning(
                    f"The engine has {len(self.queue)} requests to be processed in the queue, but none of them were added to the current batch during the execution of SyncEngine._adjust_batch"
                )

        for request_id in previous_requests_to_be_cancelled:
            if request_id not in self.requests_to_be_cancelled:
                # TODO(masahi): Need a mapping from a request ID to num_sequences
                # But for a cancelled request, it is probably enough to return only
                # one empty sequence.
                num_sequences = 1
                outputs.append(
                    RequestOutput(
                        request_id=request_id,
                        sequences=[
                            SequenceOutput(i, finish_reason=FinishReason.Cancelled)
                            for i in range(num_sequences)
                        ],
                    )
                )

        if not self.current_batch:
            return InferenceStepResult(outputs)

        requests = self._get_requests_to_process()
        results = self.text_generator.generate(requests, self.cache_manager.get_cache())
        logger.debug("Finished text generation.")

        valid_results = []
        failed_requests = set()

        for res in results:
            request_id = res.sequence_id.request_id
            # Report an error for a request if any of its generation sequences fails.
            if res.error is not None and request_id not in failed_requests:
                failed_requests.add(request_id)
                self.cache_manager.free_request(self.current_batch[request_id])
                del self.current_batch[request_id]
                outputs.append(
                    RequestOutput(
                        request_id,
                        sequences=[],
                        error=res.error,
                    )
                )
            else:
                valid_results.append(res)

        seq_outputs = defaultdict(list)

        for res in valid_results:
            request_id = res.sequence_id.request_id
            seq_index = res.sequence_id.sequence_index
            state = self.current_batch[request_id]
            gen_seq = state.generation_sequences[seq_index]
            gen_seq.next_start_position = len(gen_seq.generated_token_ids)
            new_token_ids = res.generated_tokens

            for i, token_id in enumerate(new_token_ids):
                if (
                    token_id == self.tokenizer.eos_token_id
                    and not state.debug_options.ignore_eos
                ):
                    new_token_ids = new_token_ids[:i]
                    gen_seq.is_finished = True
                    break

            gen_seq.generated_token_ids.extend(new_token_ids)

            delta = self._decode_last_output(state.prompt_token_ids, gen_seq)
            gen_seq.output_text += delta

            gen_seq.output_text, delta, gen_seq.is_finished = check_stopping_sequences(
                state.stopping_criteria, gen_seq.output_text, delta, gen_seq.is_finished
            )

            seq_outputs[request_id].append(
                SequenceOutput(
                    gen_seq.seq_index,
                    delta=delta,
                    num_generated_tokens=(len(gen_seq.generated_token_ids)),
                )
            )

        for request_id, seq_outputs in seq_outputs.items():
            state = self.current_batch[request_id]
            outputs.append(
                RequestOutput(
                    request_id,
                    sequences=seq_outputs,
                    num_prompt_tokens=state.prompt_len,
                )
            )

        logger.debug("Finished detokenization and output object creation.")

        return InferenceStepResult(outputs)

    def _adjust_batch(self):
        with self.queue_lock:
            for request_id in list(self.requests_to_be_cancelled):
                if request_id in self.current_batch:
                    state = self.current_batch.pop(request_id)
                    self.cache_manager.free_request(state)
                    self.requests_to_be_cancelled.remove(request_id)

            while self.cache_manager.get_max_new_tokens() < 1:
                request_to_remove = min(
                    self.current_batch.values(), key=lambda s: s.num_total_tokens
                )
                state = self.current_batch[request_to_remove.request_id]

                # TODO parallel sampling: Properly support Evicting a multi-sequence request
                assert state.num_sequences == 1, "Evicting a multi-sequence request is not supported."

                self.cache_manager.free_request(state)
                del self.current_batch[request_to_remove.request_id]
                self.queue.appendleft(request_to_remove)

            self._discard_cancelled_requests_from_queue()

            if self.cache_manager.get_max_new_tokens() <= self.max_decode_steps:
                logger.debug(
                    "Skip growing the batch due to max_decode_steps. Decode steps: %s",
                    self.cache_manager.get_max_new_tokens(),
                )
                return

            num_new_batched_tokens = len(self.current_batch)

            while self.queue:
                max_new_tokens = self.cache_manager.get_max_new_tokens()
                if max_new_tokens < self.min_decode_steps:
                    logger.debug(
                        "Stop growing the batch due to min_decode_steps. Decode steps: %s",
                        max_new_tokens,
                    )
                    # stop adding request if there isn't enough space to do a certain steps of decoding.
                    break

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
                        not len(self.current_batch)
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
                    logger.debug(
                        "Stop growing the batch due to max_num_batched_tokens. Batched tokens: %s",
                        num_new_batched_tokens,
                    )
                    break

                # We make sure that the KV cache will have enough free space for this request to proceed
                # decoding for at least self.max_decode_steps steps.
                if (self.cache_manager.get_free_space() - num_tokens) / (
                    len(self.current_batch) + 1
                ) < self.max_decode_steps * state.num_sequences:
                    logger.debug(
                        "Stop growing the batch due to not enough free space. Free: %s, Num tokens: %s",
                        self.cache_manager.get_free_space(),
                        num_tokens,
                    )
                    break

                self.queue.popleft()
                # TODO parallel sampling: Need update here when evicting multi-sequence requests is supported.
                self.cache_manager.allocate(state.request_id, num_tokens)
                self.current_batch[state.request_id] = state

                self._discard_cancelled_requests_from_queue()

    def _get_requests_to_process(self):
        requests = []
        # TODO: consider having hybrid batch if the underlying attention kernel supports
        # mixing prefill and decode.
        is_prompt_batch = any(
            state.generation_sequences[0].next_start_position == 0
            for state in self.current_batch.values()
        )

        if is_prompt_batch:
            for state in self.current_batch.values():
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
            for state in self.current_batch.values():
                for gen_seq in state.generation_sequences:
                    if not gen_seq.is_finished:
                        requests.append(
                            DecodeRequest(
                                sequence_id=gen_seq.seq_id,
                                token_ids=state.prompt_token_ids
                                + gen_seq.generated_token_ids,
                                sampling_params=state.sampling_params,
                            )
                        )
                        self.cache_manager.extend(
                            gen_seq.seq_id,
                            len(gen_seq.generated_token_ids)
                            - gen_seq.next_start_position,
                        )
            logger.debug("Creating decode batch with %s requests.", len(requests))

        return requests

    def has_pending_requests(self) -> bool:
        return bool(self.queue or self.current_batch)

    def _discard_cancelled_requests_from_queue(self):
        """
        Requires the self.queue_lock to be held before calling this function.
        """
        while self.queue and self.queue[0].request_id in self.requests_to_be_cancelled:
            state = self.queue.popleft()
            self.requests_to_be_cancelled.remove(state.request_id)

    def _get_new_request_state(self, request: Request) -> RequestState:
        if request.debug_options.prompt is not None:
            prompt = request.debug_options.prompt
        else:
            prompt = self.conversation_template.apply(request.messages)

        prompt_tokens = self.tokenizer.encode(prompt)

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

    def _decode_last_output(
        self, prompt_tokens: list[int], generation_sequence: GenerationSequence
    ) -> str:
        if len(generation_sequence.output_text):
            prefix_idx = max(0, generation_sequence.next_start_position - 6)
        else:
            prefix_idx = generation_sequence.next_start_position

        token_ids = prompt_tokens + generation_sequence.generated_token_ids

        if prefix_idx == 0:
            return self.tokenizer.decode(token_ids)

        prefix = self.tokenizer.decode(
            token_ids[prefix_idx : generation_sequence.next_start_position]
        )
        full = self.tokenizer.decode(token_ids[prefix_idx:])

        return full[len(prefix) :]

    def _should_stop_by_length(self, state: RequestState) -> bool:
        # TODO: currently, we simply return true for both stopping reasons.
        #       in the future, we can differentiate these two.
        # this include prompt tokens and gen tokens so far
        for gen_seq in state.generation_sequences:
            if gen_seq.is_finished:
                continue

            num_context_tokens = len(
                state.prompt_token_ids + gen_seq.generated_token_ids
            )
            if num_context_tokens < self.max_context_length:
                return False
            num_gen_tokens = num_context_tokens - state.prompt_len
            if (
                state.stopping_criteria.max_tokens is not None
                and num_gen_tokens < state.stopping_criteria.max_tokens
            ):
                return False

        return True
