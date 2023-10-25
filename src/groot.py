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

from src.cache import Cache

ASCII_ART = """
 {v':   `   / `&~-,
 v .              v
 V  .~.      .~.  V
 : (  0)    (  0) :
  i `'`      `'` j
   i     __    ,j
    `%`~....~'&
   o' /  /` -S,
  o.~'.  )(  r  .o ,.   Dims: <dims>
 o',  %``/``& : 'bF     Loss: <loss>
d', ,ri.~~-~.ri , +h    Nodes: <nodes>
`oso' d`~..~`b 'sos`    Depth: <depth>
     d`+ II +`b
     i_:_yi_;y          GROoT v0.1
"""


class GROoT:
    """Guided Recursive Optimization over Tree"""

    def __init__(
        self,
        dims: int, branch_factor: int,
        initial_loc: Optional[float] = 1.0, initial_scale: Optional[float] = 0.2,
        decrease_factor: Optional[float] = 0.5,
        dtype: np.dtype = np.float32,
        cache: Optional[Cache] = None
    ):
        self.dims = dims
        self.branch_factor = branch_factor
        self.initial_loc = initial_loc
        self.initial_scale = initial_scale
        self.decrease_factor = decrease_factor
        self.dtype = dtype
        self.cache = cache

        self.nodes: Dict[str, Node] = {
            'Origin': Origin(self.dims, self.dtype)
        }

        self.sorted_nodes = []

    def __str__(self):
        art = ASCII_ART
        art = art.replace('<dims>', str(self.dims))
        art = art.replace('<loss>', str(
            min(map(lambda n: n.loss, self.sorted_nodes))))
        art = art.replace('<nodes>', str(len(self.sorted_nodes)))
        art = art.replace('<depth>', str(
            max(map(lambda n: n.depth, self.sorted_nodes))))
        return art

    def create_nodes(self, branch_factor: Optional[int] = None):
        branch_factor = branch_factor or self.branch_factor

        if not self.sorted_nodes:
            new_branches = [
                Node(
                    self.nodes['Origin'],
                    uuid.uuid4(),
                    self.initial_loc,
                    self.initial_scale,
                    self.dtype,
                    self.cache
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
                self.dtype,
                self.cache
            )
            new_branches.append(new_node)
        return new_branches

    def sample_node(self):
        # We want to explore nodes that have the lowest loss
        weights = np.array(list(map(lambda n: 1/n.loss, self.sorted_nodes)))
        # We take into account the depth of the node, since it's acceptable for shallow nodes to have high loss
        depths = np.array(list(map(lambda n: n.depth, self.sorted_nodes)))
        weights /= depths / depths.max()
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
        self.depth = 0

    def get_position(self):
        return np.zeros(shape=self.dims, dtype=self.dtype)


class Node:
    def __init__(
        self,
        parent: "Node",
        uuid: uuid.UUID,
        loc: float,
        scale: float,
        dtype: np.dtype = np.float32,
        cache: Optional[Cache] = None
    ):
        self.parent = parent
        self.uuid = uuid
        self.loc = loc
        self.scale = scale
        self.dtype = dtype
        self.cache = cache

        self.__loss: Optional[float] = None
        self.__depth = None

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, value):
        if self.loss is not None:
            raise Exception('Loss already set. Cannot change loss of a node')
        self.__loss = value

    @property
    def dims(self) -> int:
        return self.parent.dims

    @property
    def depth(self) -> int:
        if self.__depth is None:
            self.__depth = self.parent.depth + 1
        return self.__depth

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

    def get_position(self, use_cache=True):
        """absolute position of node in parameter space"""
        if not use_cache or self.cache is None:
            return self.parent.get_position() + self.get_offset()
        return self.cache[self]
