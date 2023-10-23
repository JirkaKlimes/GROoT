import matplotlib.pyplot as plt
import numpy as np

from src.groot import GROoT
from src.cache import PythonDictCache

TARGET = np.random.uniform(-1, 1, 2)

fig = plt.figure()
ax = fig.add_subplot()
groot = GROoT(2, 16, cache=PythonDictCache(1024))


while True:
    nodes = groot.create_nodes()

    for n in nodes:
        loss = np.linalg.norm(n.get_position() - TARGET)
        n.loss = loss

    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.plot(TARGET[0], TARGET[1], 'ro')
    for n in groot.sorted_nodes:
        pos1 = n.get_position()
        pos2 = n.parrent.get_position()
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'g-', linewidth=0.2)

    groot.add_rated_nodes(nodes)
    print(groot.sorted_nodes[0].loss)
    plt.pause(0.01)

plt.show()
