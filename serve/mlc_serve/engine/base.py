from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .sampling_params import SamplingParams, SamplingType

RequestId = str


@dataclass
class StoppingCriteria:
    """
    Parameters about when to stop text generation.
    """

    max_tokens: Optional[int]


@dataclass
class ChatMessage:
    role: str
    content: Optional[str]


@dataclass
class DebugOptions:
    ignore_eos: bool = False
    # Override messages with a single prompt, skipping conversation template
    prompt: Optional[str] = None


class FinishReason(Enum):
    Stop = "stop"
    Length = "length"
    Cancelled = "cancelled"


@dataclass
class Request:
    request_id: RequestId
    messages: list[ChatMessage]

    # Number of sequences to generate
    num_sequences: int = 1
    # TODO: should `best_of` be handled in the serving layer?
    best_of: int = None

    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    stopping_criteria: StoppingCriteria = field(default_factory=StoppingCriteria)
    debug_options: DebugOptions = field(default_factory=DebugOptions)

    def __post_init__(self):
        if self.best_of is None:
            self.best_of = self.num_sequences
        if self.num_sequences < 1:
            raise ValueError(
                f"num_sequences must be at least 1, got {self.num_sequences}."
            )
        if self.best_of < self.num_sequences:
            raise ValueError(
                f"best_of must be greater than or equal to num_sequences, "
                f"got n={self.num_sequences} and best_of={self.best_of}."
            )
        if (
            self.best_of > 1
            and self.sampling_params.sampling_type == SamplingType.GREEDY
        ):
            raise ValueError(
                "best_of must be 1 when using greedy sampling." f"Got {self.best_of}."
            )


@dataclass
class SequenceOutput:
    index: int
    delta: Optional[str] = None
    # If finish_reason is not None, delta should be None.
    finish_reason: Optional[FinishReason] = None

    @property
    def is_finished(self) -> bool:
        return self.finish_reason is not None


@dataclass
class RequestOutput:
    request_id: RequestId
    sequences: list[SequenceOutput]
    error: Optional[str] = None

    @property
    def is_finished(self) -> bool:
        return self.error is not None or all(seq.is_finished for seq in self.sequences)


@dataclass
class InferenceStepResult:
    outputs: list[RequestOutput]


class InferenceEngine:
    def add(self, requests: list[Request]):
        """
        Add requests to the InferenceEngine.

        Requests will be processed when `step` is called, if there is capacity.
        Requests will be handled on a first-in, first-out basis.
        """

    def cancel(self, request_id: RequestId):
        """
        Cancel the generation of a request.

        The next call to `step` will return TextGenerationOutput for cancelled requests.
        The output will contain empty delta and finish reason `cancelled`.
        """

    def wait_for_request(self, timeout_seconds=None) -> bool:
        """
        Block until there is request to process.
        It can also be given a timeout_seconds parameter, allowing it to return even if
        no requests are coming in. The return value is a boolean that indicates whether
        there are requests when it's returned.
        """

    def step(self) -> InferenceStepResult:
        """
        Perform a single inference step. In general it will generates one token for each
        requests that are being processed, but it could generate more if speculative decoding
        is used.

        If the engine has no requests in the queue, `step` will return immediately with
        an empty `InferenceStepResult.outputs`.
        """
