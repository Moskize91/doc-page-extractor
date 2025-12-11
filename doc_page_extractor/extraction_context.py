from typing import Any, Callable, cast
from transformers import StoppingCriteria

from .types import ExtractionContext


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
        self._raw_context: ExtractionContext = context
        self._check_aborted: Callable[[], bool] = context.check_aborted
        self._max_tokens: int | None = context.max_tokens
        self._max_output_tokens: int | None = context.max_output_tokens
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._error: ExtractionAbortedError | None = None

        error: TokenLimitError | None = None
        if self._max_tokens is not None:
            self._max_tokens -= context.input_tokens
            self._max_tokens -= context.output_tokens
            if self._max_tokens <= 0:
                error = TokenLimitError()
        if not error and self._max_output_tokens is not None:
            self._max_output_tokens -= context.output_tokens
            if self._max_output_tokens <= 0:
                error = TokenLimitError()
        if error:
            error.input_tokens = context.input_tokens
            error.output_tokens = context.output_tokens
            self._error = error

    def notify_finished(self):
        if self._input_tokens is not None:
            self._raw_context.input_tokens += self._input_tokens
        if self._output_tokens is not None:
            self._raw_context.output_tokens += self._output_tokens
        if self._error:
            self._error.input_tokens = self._raw_context.input_tokens
            self._error.output_tokens = self._raw_context.output_tokens
            raise self._error

    def __call__(self, input_ids, scores, **kwargs) -> Any:
        if self._error:
            return cast(Any, True)

        tokens_count: int = 0
        for i in range(input_ids.shape[0]):
            tokens_count += input_ids[i].shape[0]

        if self._input_tokens is None:
            # 首次调用在接收到第一个 output token 时，故可反推 input_tokens
            self._input_tokens = tokens_count - 1

        self._output_tokens = tokens_count - self._input_tokens

        if (self._max_tokens is not None and tokens_count > self._max_tokens) or (
            self._max_output_tokens is not None
            and self._output_tokens > self._max_output_tokens
        ):
            self._error = TokenLimitError()
            return cast(Any, True)

        if self._check_aborted():
            self._error = AbortError()
            return cast(Any, True)

        return cast(Any, False)
