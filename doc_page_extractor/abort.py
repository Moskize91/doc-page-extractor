from dataclasses import dataclass
from typing import Any, Callable, cast

import torch
from transformers import StoppingCriteria


@dataclass
class AbortContext:
    check_aborted: Callable[[], bool]
    max_new_tokens: int | None = None
    no_repeat_ngram_size: int | None = None

class AbortError(Exception):
    pass

class AbortStoppingCriteria(StoppingCriteria):
    def __init__(self, context: AbortContext) -> None:
        super().__init__()
        self._aborted: bool = False
        self._check_aborted: Callable[[], bool] = context.check_aborted

    @property
    def aborted(self) -> bool:
        return self._aborted

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        is_aborted: bool
        if self._aborted:
            is_aborted = True
        else:
            self._aborted = self._check_aborted()
            is_aborted = self._aborted
        return cast(Any, is_aborted)
