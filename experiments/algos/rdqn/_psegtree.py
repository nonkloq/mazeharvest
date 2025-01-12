"""Slightly slower than the thread version"""

import random
from multiprocessing import Event, Process, Queue
from multiprocessing.shared_memory import SharedMemory
from queue import Empty

import numpy as np


# There is no lock to handle access of shared memory buffers
# because the update & sample buffer logic is gauranteed to be in sequence
# so prefetch will only be called after the get_indexes in sample() method
def updater(
    # lock: Lock,
    queue: Queue,
    stop: Event,
    done: Event,
):
    """Segment Tree Updater Process, Updates & Retrieve indexes in parallel."""

    # Accessing the shared Memories by name and using them as np array
    sm1 = SharedMemory("sum_tree")
    arr_len = sm1.size // np.dtype(np.float64).itemsize
    capacity = arr_len // 2
    sum_tree = np.ndarray(arr_len, dtype=np.float64, buffer=sm1.buf)

    sm2 = SharedMemory("min_tree")
    min_tree = np.ndarray(arr_len, dtype=np.float64, buffer=sm2.buf)

    sm3 = SharedMemory("requested_idxs")
    batchsize = sm3.size // np.dtype(np.int64).itemsize
    requested_idxs = np.ndarray(batchsize, dtype=np.int64, buffer=sm3.buf)

    def _update_segtrees(idx: int, priority_alpha: float, capacity: int):
        """Update the values for the idx and update the segtrees"""
        idx += capacity
        min_tree[idx] = priority_alpha
        sum_tree[idx] = priority_alpha

        while idx >= 2:
            idx //= 2
            left = 2 * idx
            right = left + 1
            sum_tree[idx] = sum_tree[left] + sum_tree[right]
            min_tree[idx] = min(min_tree[left], min_tree[right])

    def _retreive_index(ridx: int, upperbound: float, limit: int, batchsize: int):
        """Find the largest i such that sum_{k=1}^{i} p_k^alpha <= p"""
        idx = 1
        while idx < limit:
            left = idx * 2
            # Check if left branch sum is > than upperbound
            if sum_tree[left] > upperbound:
                # Go to left to find sum smaller than the upperbound
                idx = left
            else:
                # If the left sum <= upperbound
                # Go to right branch and reduce the upperbound by sum from left
                upperbound -= sum_tree[left]
                idx = left + 1  # Right node index

        requested_idxs[ridx] = idx - limit
        # Mark the whole retreival taks as Done
        if ridx == (batchsize - 1):
            done.set()

    while not stop.is_set():
        try:
            task, data = queue.get(timeout=0.1)

            match task:
                case 0:
                    # with lock:
                    _update_segtrees(*data, capacity)
                case 1:
                    # with lock:
                    _retreive_index(*data, batchsize)
                case _:
                    raise ValueError("Unkown task.")
        except Empty:
            continue

    for sm in (sm1, sm2, sm3):
        sm.close()


class SegmentTreeSM:
    def __init__(self, capacity: int, batchsize: int):
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."

        self.capacity = capacity
        self.__batchsize = batchsize

        # Creating Shared Memory Buffers
        self.__shared_memories = (
            SharedMemory(
                name="sum_tree",
                create=True,
                size=np.dtype(np.float64).itemsize * self.capacity * 2,
            ),
            SharedMemory(
                name="min_tree",
                create=True,
                size=np.dtype(np.float64).itemsize * self.capacity * 2,
            ),
            SharedMemory(
                name="requested_idxs",
                create=True,
                size=np.dtype(np.int64).itemsize * batchsize,
            ),
        )

        # Data Access Lock
        # self._lock = Lock()

        self.sum_tree = np.ndarray(
            2 * self.capacity, dtype=np.float64, buffer=self.__shared_memories[0].buf
        )
        self.min_tree = np.ndarray(
            2 * self.capacity, dtype=np.float64, buffer=self.__shared_memories[1].buf
        )
        self.__requested_idxs = np.ndarray(
            batchsize, dtype=np.int64, buffer=self.__shared_memories[2].buf
        )
        # Intializing the values
        self.sum_tree[:] = 0
        self.min_tree[:] = float("inf")
        self.__requested_idxs[:] = -1

        # Task queue
        self.__task_queue = Queue()

        # Events
        self.__stop_event = Event()
        self.__pres_event = Event()  # Presample indicator
        self.__done_event = Event()

        # Starting the Updater Process
        self.__uproc = Process(
            target=updater,
            args=(
                # self._lock,
                self.__task_queue,
                self.__stop_event,
                self.__done_event,
            ),
            daemon=True,
        )

        self.__uproc.start()

    def get_indexes(self, timeout: float = 1.0) -> np.ndarray:
        if not self.__pres_event.is_set():
            raise Exception(
                "There is no previous presample request to get indexes from."
            )

        if self.__done_event.wait(timeout):
            self.__pres_event.clear()
            return self.__requested_idxs  # .copy()

        raise Exception(
            f"Possible Dead Lock?? Can't able to complete the request in {timeout} seconds."
        )

    def send_sample_request(self, lim: int):
        """
        Checks for any presample request, if there any, just return.
        otherwise initiate a presample request so that in buffer sample()
        method get_indexes method can be used to retrieve indices
        """
        if self.__pres_event.is_set():
            return

        self.send_presample_request(lim)

    def send_presample_request(self, lim: int):
        if self.__pres_event.is_set():
            raise Exception(
                "Please Retrieve the existing request using `get_indexes()` method before sending a new one."
            )

        self.__pres_event.set()
        self.__done_event.clear()

        p_total = self._sum
        segment = p_total / self.__batchsize
        for i in range(self.__batchsize):
            lo = segment * i
            hi = segment * (i + 1)
            upperbound = random.uniform(lo, hi)
            # or we can just use random.random() instead of equally
            # segmenting the p_total
            # upperbound = random.random() * p_total
            self.__task_queue.put((1, (i, upperbound, lim)))

    @property
    def _sum(self):
        """sum_k p_k^alpha"""
        return self.sum_tree[1]

    @property
    def _min(self):
        """min_k p_k^alpha"""
        return self.min_tree[1]

    def __setitem__(self, idx: int, priority_alpha: float):
        """Adds the segment tree update request in queue"""
        self.__task_queue.put((0, (idx, priority_alpha)))

    def __getitem__(self, idx: int | np.ndarray) -> float:
        """Get real value(s) in leaf node of sum_tree."""
        return self.sum_tree[idx + self.capacity]

    def __del__(self):
        self.__stop_event.set()
        self.__uproc.join(1.0)
        for sm in self.__shared_memories:
            sm.close()
            sm.unlink()

        self.__uproc.close()
        del self.min_tree
        del self.sum_tree
