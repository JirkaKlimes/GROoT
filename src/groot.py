from typing import Optional, Dict, List
import uuid
import os
if os.getenv('DEVICE') == 'CUDA':
    print('Using CUDA for math operations')
    import cupy as np
else:
    import numpy as np
import bisect
import random


class GROoT:
    """Guided Recursive Optimization over Tree"""

    def __init__(
        self,
        dims: int, branch_factor: int,
        initial_loc: Optional[float] = 1.0, initial_scale: Optional[float] = 0.2,
        decrease_factor: Optional[float] = 0.5,
        dtype: np.dtype = np.float32
    ):
        self.dims = dims
        self.branch_factor = branch_factor
        self.initial_loc = initial_loc
        self.initial_scale = initial_scale
        self.decrease_factor = decrease_factor
        self.dtype = dtype

        self.nodes: Dict[str, Node] = {
            'Origin': Origin(self.dims, self.dtype)
        }

        self.sorted_nodes = []

    def create_nodes(self, branch_factor: Optional[int] = None):
        branch_factor = branch_factor or self.branch_factor

        if not self.sorted_nodes:
            new_branches = [
                Node(
                    self.nodes['Origin'],
                    uuid.uuid4(),
                    self.initial_loc,
                    self.initial_scale,
                    self.dtype
                )
                for _ in range(branch_factor)
            ]
            return new_branches

        new_branches = []
        for _ in range(branch_factor):
            node = self.sample_node()
            new_node = Node(
                node,
                uuid.uuid4(),
                node.loc * self.decrease_factor,
                node.scale * self.decrease_factor,
                self.dtype
            )
            new_branches.append(new_node)
        return new_branches

    def sample_node(self):
        weights = np.array(list(map(lambda n: 1/n.loss, self.sorted_nodes)))
        probabilities = weights / weights.sum()
        return random.choices(self.sorted_nodes, probabilities)[0]

    def add_node(self, node: "Node"):
        self.nodes[str(node.uuid)] = node
        i = bisect.bisect(self.sorted_nodes, node)
        self.sorted_nodes.insert(i, node)

    def add_rated_nodes(self, nodes: List["Node"]):
        for node in nodes:
            if node.loss is None:
                raise Exception("Trying to add node without loss!")
            self.add_node(node)


class Origin:
    def __init__(self, dims: int, dtype: np.dtype = np.float32):
        self.dims = dims
        self.dtype = dtype

    def get_position(self):
        return np.zeros(shape=self.dims, dtype=self.dtype)


class Node:
    def __init__(self, parent: "Node", uuid: uuid.UUID, loc: float, scale: float, dtype: np.dtype = np.float32):
        self.parrent = parent
        self.uuid = uuid
        self.loc = loc
        self.scale = scale
        self.dtype = dtype
        self.loss: Optional[float] = None

    @property
    def dims(self) -> int:
        return self.parrent.dims

    def __lt__(self, other: "Node") -> bool:
        return self.loss < other.loss

    def rng_generator(self) -> np.random.Generator:
        """Create RNG from UUID"""
        seed = np.frombuffer(bytes.fromhex(self.uuid.hex), np.uint32)
        return np.random.default_rng(seed)

    def get_offset(self):
        """offset of node realative to parent"""
        rng = self.rng_generator()
        # sample random point on a nd-sphere
        point = rng.normal(size=self.dims)
        point /= np.linalg.norm(point)
        # scale it by some random number
        offset = point * rng.normal(loc=self.loc, scale=self.scale)
        return offset

    def get_position(self):
        """absolute position of node in parameter space"""
        # This needs to use caching for higher dimensions, otherwise not usable
        return self.parrent.get_position() + self.get_offset()
