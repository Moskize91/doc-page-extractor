from dataclasses import dataclass
from typing import Any, Callable, cast

import torch
from transformers import StoppingCriteria


@dataclass
class AbortContext:
    check_aborted: Callable[[], bool]
    max_new_tokens: int | None = None
    no_repeat_ngram_size: int | None = None


class AbortStoppingCriteria(StoppingCriteria):
    def __init__(self, context: AbortContext) -> None:
        super().__init__()
        self._check_aborted: Callable[[], bool] = context.check_aborted

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        return cast(Any, self._check_aborted())
