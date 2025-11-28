from typing import cast,Callable, TypeVar
from threading import Lock


T = TypeVar("T")


LazyGetter = Callable[[], T]

def lazy_load(load: Callable[[], T]) -> LazyGetter[T]:
    lock = Lock()
    did_load = False
    value: T | None = None

    def getter() -> T:
        nonlocal did_load, value
        with lock:
            if not did_load:
                value = load()
                did_load = True
            return cast(T, value)
    return getter