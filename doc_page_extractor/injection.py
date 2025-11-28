"""
Model Inference Interruption Injection

This module provides a context manager to inject interruption capabilities into
DeepSeek-OCR model's infer() method via monkey patching.

WHY WE NEED THIS HACK:
----------------------
1. DeepSeek-OCR's model.infer() is a time-consuming operation (can take seconds to minutes)
2. The infer() method internally calls self.generate() from transformers library
3. transformers.generate() supports stopping_criteria for interruption control
4. However, DeepSeek-OCR's infer() method does NOT expose this parameter
5. The model code is downloaded from HuggingFace Hub with trust_remote_code=True
6. Modifying cached files would break when model updates, so we use runtime injection

APPROACH:
---------
We replace the model's generate() method ONCE at initialization time with a wrapper
that reads stopping_criteria from thread-local storage. This allows:
- Thread-safe concurrent inference
- Clean interruption via StoppingCriteria interface
- Timeout control for long-running inference
- User-triggered cancellation
- No modification to downloaded model files
- Automatic compatibility with model updates

THREAD SAFETY:
--------------
This implementation uses thread-local storage to pass context between threads.
The model's generate() method is patched only once during initialization,
making it safe for concurrent use by multiple threads.

USAGE:
------
    from doc_page_extractor.injection import preprocess_model, InferWithInterruption

    # Step 1: Preprocess model once at initialization
    model = AutoModel.from_pretrained(...)
    preprocess_model(model)

    # Step 2: Use in concurrent inference
    with InferWithInterruption(context=context) as infer:
        result = infer(
            model,
            tokenizer,
            prompt="<image>\\n<|grounding|>Convert the document to markdown.",
            image_file="input.png",
            output_path="./output",
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True,
            eval_mode=True
        )
"""

import threading
from typing import Any

from transformers import StoppingCriteria

from .types import ExtractionContext
from .extraction_context import AbortStoppingCriteria

_LOCAL = threading.local()
_LOCAL_KEY = "value"


def preprocess_model(model: Any) -> Any:
    original_generate = model.generate

    def thread_safe_generate(*args, **kwargs):
        stopping_criteria = getattr(_LOCAL, _LOCAL_KEY, None)
        if stopping_criteria is not None:
            stopping: list[StoppingCriteria] = kwargs.get("stopping_criteria", [])
            stopping.append(stopping_criteria)
            kwargs["stopping_criteria"] = stopping
        return original_generate(*args, **kwargs)

    model.generate = thread_safe_generate
    return model


class InferWithInterruption:
    def __init__(
        self,
        model: Any,
        context: ExtractionContext | None,
    ):
        self._model = model
        self._stopping: AbortStoppingCriteria | None = None
        if context:
            self._stopping = AbortStoppingCriteria(context)

    def __enter__(self):
        setattr(_LOCAL, _LOCAL_KEY, self._stopping)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(_LOCAL, _LOCAL_KEY, None)
        return False

    def __call__(self, *args, **kwargs):
        """Direct call to model.infer()"""
        result = self._model.infer(*args, **kwargs)
        if self._stopping:
            self._stopping.notify_finished()
        return result
