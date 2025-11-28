import threading
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Node(Generic[T]):
    resource: T
    lock: threading.Lock


class ResourceLock(Generic[T]):
    def __init__(self, resource: T, lock: threading.Lock) -> None:
        self._resource = resource
        self._lock = lock

    @property
    def resource(self) -> T:
        return self._resource

    def __enter__(self) -> T:
        self._lock.acquire()
        return self._resource

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self._lock.release()


class ResourceLocks(Generic[T]):
    def __init__(self, resources: list[T]) -> None:
        if not resources:
            raise ValueError("resources must not be empty")

        self._nodes = [_Node(resource=r, lock=threading.Lock()) for r in resources]
        self._next_index = 0
        self._index_lock = threading.Lock()

    def access(self) -> ResourceLock[T]:
        # TODO: 这是个简单的轮询逻辑，无法做到先到先得（只能基本做到）有优化空间
        with self._index_lock:
            start_index = self._next_index
            self._next_index = (self._next_index + 1) % len(self._nodes)

        for offset in range(len(self._nodes)):
            index = (start_index + offset) % len(self._nodes)
            node = self._nodes[index]

            if node.lock.acquire(blocking=False):
                return ResourceLock(node.resource, node.lock)

        node = self._nodes[start_index]
        node.lock.acquire()
        return ResourceLock(node.resource, node.lock)
