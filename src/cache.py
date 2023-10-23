from typing import List, Dict, TYPE_CHECKING
import bisect
import numpy as np
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from src.groot import Node


class Cache(ABC):
    @ abstractmethod
    def __getitem__(self, node: "Node") -> np.ndarray:
        raise NotImplementedError


class PythonDictCache(Cache):
    def __init__(self, size: int) -> None:
        self.size = size

        self.cache: Dict[str, np.ndarray] = {}
        self.sorted_cache: List["Node"] = []

    def __getitem__(self, node: "Node"):
        # If the node is already in the cache, return its value
        if node.uuid in self.cache:
            return self.cache[node.uuid]

        # If we don't know the node's loss or the sorted_cache is not empty and the last node has no loss,
        # we cannot make a decision to cache it, so compute the value without using the cache.
        if node.loss is None or (self.sorted_cache and (self.sorted_cache[-1].loss is None)):
            return node.get_position(use_cache=False)

        # If the cache is full
        if len(self.sorted_cache) >= self.size:
            # Check if the node's loss is higher than the worst node in the cache.
            if node.loss > self.sorted_cache[-1].loss:
                # If yes, compute the value without using the cache.
                return node.get_position(use_cache=False)

            # If the node is worth saving, remove the worst node from the cache.
            self.sorted_cache.pop()

        # Add the node to the cache while maintaining the sorted order.
        i = bisect.bisect(self.sorted_cache, node)
        self.sorted_cache.insert(i, node)
        # Cache the node's value.
        self.cache[node.uuid] = node.get_position(use_cache=False)
        return self.cache[node.uuid]  # Return the cached value.
