import queue
import random
from threading import Event, Thread

import numpy as np


class SegmentTreeSM:
    """
    Fast SegmentTree (Sum & Min)
        uses daemon threads to update the segment trees
        faster for buffersize that are larger than 2^12.
    Attributes:
        capacity (int)
        batchsize (int)
    """

    def __init__(self, capacity: int, batchsize: int):
        """Initialization.

        Args:
            capacity (int)
            batchsize (int)
        """

        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."

        self.capacity = capacity
        self.__batchsize = batchsize

        self.sum_tree = np.zeros(2 * self.capacity, dtype=np.float64)
        self.min_tree = np.full(2 * self.capacity, float("inf"), dtype=np.float64)

        self._uthread = Thread(target=self._updater, daemon=True)

        # Task queue
        self.__task_queue = queue.Queue()

        # Events
        self.__stop_event = Event()
        self.__pres_event = Event()  # Presample indicator
        self.__done_event = Event()

        self.__requested_idxs = np.zeros(batchsize, dtype=np.int64)
        self._uthread.start()

    def get_indexes(self, timeout: float = 1.0) -> np.ndarray:
        if not self.__pres_event.is_set():
            raise Exception(
                "There is no previous presample request to get indexes from."
            )

        if self.__done_event.wait(timeout):
            self.__pres_event.clear()
            return self.__requested_idxs

        raise Exception(
            f"Possible Dead Lock?? Can't able to complete the request in {timeout} seconds."
        )

    def send_sample_request(self, lim: int) -> bool:
        """
        Checks for any presample request, if there any, just return.
        otherwise initiate a presample request so that in buffer sample()
        method get_indexes method can be used to retrieve indices
        """
        if self.__pres_event.is_set():
            return

        self.send_presample_request(lim)

    # There is no lock to handle access of __requested_idxs variable
    # because the update & sample buffer logic is gauranteed to be in sequence
    # so prefetch will only be called after the get_indexes in sample() method
    # Here the array can only be created by this method and only be set to none
    # by the get_indexes() method
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

    def _update_segtrees(self, idx: int, priority_alpha: float):
        """Update the values for the idx and update the segtrees"""
        idx += self.capacity

        self.min_tree[idx] = priority_alpha
        self.sum_tree[idx] = priority_alpha

        while idx >= 2:
            idx //= 2
            left = 2 * idx
            right = left + 1
            self.min_tree[idx] = min(self.min_tree[left], self.min_tree[right])
            self.sum_tree[idx] = self.sum_tree[left] + self.sum_tree[right]

    def _retrieve_index(self, ridx: int, upperbound: float, limit: int):
        """Find the largest i such that sum_{k=1}^{i} p_k^alpha <= p"""
        idx = 1
        while idx < limit:
            left = idx * 2
            # Check if left branch sum is > than upperbound
            if self.sum_tree[left] > upperbound:
                # Go to left to find sum smaller than the upperbound
                idx = left
            else:
                # If the left sum <= upperbound
                # Go to right branch and reduce the upperbound by sum from left
                upperbound -= self.sum_tree[left]
                idx = left + 1  # Right node index

        self.__requested_idxs[ridx] = idx - limit
        if ridx == (self.__batchsize - 1):
            self.__done_event.set()

    def _updater(self):
        """Updates Sum & Min SegmentTrees in a Seperate Thread"""
        while not self.__stop_event.is_set():
            try:
                task, data = self.__task_queue.get(timeout=0.1)
                match task:
                    case 0:
                        self._update_segtrees(*data)
                    case 1:
                        self._retrieve_index(*data)
                    case _:
                        raise Exception("Unknown task.")

                # self._task_queue.task_done()
            except queue.Empty:
                continue

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
        del self.min_tree
        del self.sum_tree
        self.__stop_event.set()
        self._uthread.join(1.0)
