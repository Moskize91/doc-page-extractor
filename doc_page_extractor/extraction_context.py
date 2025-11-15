from dataclasses import dataclass
from typing import Any, Callable, cast

import torch
from transformers import StoppingCriteria


@dataclass
class ExtractionContext:
    check_aborted: Callable[[], bool]
    max_tokens: int | None = None
    max_output_tokens: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class ExtractionAbortedError(Exception):
    def __init__(self):
        super().__init__("Extraction was aborted.")
        self.input_tokens: int = 0
        self.output_tokens: int = 0


class AbortError(ExtractionAbortedError):
    pass


class TokenLimitError(ExtractionAbortedError):
    pass


class AbortStoppingCriteria(StoppingCriteria):
    def __init__(self, context: ExtractionContext) -> None:
        super().__init__()
        self._context: ExtractionContext = context
        self._input_tokens: int | None = None
        self._error: ExtractionAbortedError | None = None

    def raise_if_error(self):
        if self._error:
            raise self._error

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        if self._error:
            return cast(Any, True)

        tokens_count: int = 0
        for i in range(input_ids.shape[0]):
            tokens_count += input_ids[i].shape[0]

        if self._input_tokens is None:
            # 首次调用在接收到第一个 output token 时，故可反推 input_tokens
            self._input_tokens = tokens_count - 1
            self._context.input_tokens = self._input_tokens

        output_tokens = tokens_count - self._input_tokens
        self._context.output_tokens = output_tokens

        if (
            self._context.max_tokens is not None
            and tokens_count >= self._context.max_tokens
        ) or (
            self._context.max_output_tokens is not None
            and output_tokens >= self._context.max_output_tokens
        ):
            self._error = TokenLimitError()
            return cast(Any, True)

        if self._context.check_aborted():
            self._error = AbortError()
            return cast(Any, True)

        return cast(Any, False)
