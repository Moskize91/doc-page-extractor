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
We temporarily replace the model's generate() method to inject stopping_criteria,
then restore it after inference completes. This allows:
- Clean interruption via StoppingCriteria interface
- Timeout control for long-running inference
- User-triggered cancellation
- No modification to downloaded model files
- Automatic compatibility with model updates

THREAD SAFETY:
--------------
This implementation uses a lock to ensure thread-safe patching. Since GPU resources
are limited and the model is large, concurrent inference is not practical anyway.

LIMITATIONS:
------------
1. If exceptions occur during inference, the lock will be released but errors will propagate
2. Stack traces may be slightly less clear due to method wrapping
3. If DeepSeek-OCR changes their infer() implementation significantly, this may break
   (though it will fail loudly rather than silently)

USAGE:
------
    from doc_page_extractor.injection import InferWithInterruption
    from transformers import MaxTimeCriteria, StoppingCriteria
    import threading

    # Example 1: With timeout
    with InferWithInterruption(model, stopping_criteria=[MaxTimeCriteria(60.0)]) as infer:
        result = infer(
            tokenizer=tokenizer,
            prompt="<image>\\n<|grounding|>Convert the document to markdown.",
            image_file="input.png",
            output_path="./output",
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True,
            eval_mode=True
        )

    # Example 2: With manual cancellation
    class CancelCriteria(StoppingCriteria):
        def __init__(self):
            self.event = threading.Event()
        def cancel(self):
            self.event.set()
        def __call__(self, input_ids, scores, **kwargs):
            return self.event.is_set()

    cancel_criteria = CancelCriteria()
    with InferWithInterruption(model, stopping_criteria=[cancel_criteria]) as infer:
        # In another thread, you can call: cancel_criteria.cancel()
        result = infer(...)
"""

from typing import Any, Callable, cast

from transformers import StoppingCriteria

from .extraction_context import AbortStoppingCriteria, ExtractionContext


class InferWithInterruption:
    def __init__(
        self,
        model: Any,
        context: ExtractionContext | None,
    ):
        self._model = model
        self._stopping: AbortStoppingCriteria | None = None
        self._original_generate: Callable | None = None
        if context:
            self._stopping = AbortStoppingCriteria(context)

    def __enter__(self) -> Callable:
        self._original_generate = self._model.generate

        def patched_generate(*args, **kwargs):
            if self._stopping:
                stopping: list[StoppingCriteria] = kwargs.get("stopping_criteria", [])
                stopping.append(self._stopping)
                kwargs["stopping_criteria"] = stopping
            return cast(Callable, self._original_generate)(*args, **kwargs)

        self._model.generate = patched_generate
        return self._proxy_infer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_generate is not None:
            self._model.generate = self._original_generate
            self._original_generate = None
        return False

    def _proxy_infer(self, *args, **kwargs):
        result = self._model.infer(*args, **kwargs)
        if self._stopping:
            self._stopping.notify_finished()
        return result
