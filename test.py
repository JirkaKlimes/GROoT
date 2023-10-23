import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from src.groot import GROoT
from src.cache import PythonDictCache

TARGET = np.random.uniform(-1, 1, 2)

fig = plt.figure()
tree_ax = fig.add_subplot(1, 2, 1)
loss_ax = fig.add_subplot(1, 2, 2)
groot = GROoT(2, 128, cache=PythonDictCache(1024))


while True:
    nodes = groot.create_nodes()

    for n in nodes:
        loss = np.linalg.norm(n.get_position() - TARGET)
        n.loss = loss

    groot.add_rated_nodes(nodes)

    loss_ax.cla()
    losses = list(map(lambda n: n.loss, groot.sorted_nodes))
    kde = gaussian_kde(losses)
    x = np.linspace(max(losses), min(losses), 100)
    y = kde(x)
    loss_ax.plot(x, y)

    tree_ax.set_xlim(-1, 1)
    tree_ax.set_ylim(-1, 1)
    tree_ax.plot(TARGET[0], TARGET[1], 'ro')
    for n in nodes:
        pos1 = n.get_position()
        pos2 = n.parrent.get_position()
        tree_ax.plot([pos1[0], pos2[0]], [
                     pos1[1], pos2[1]], 'g-', linewidth=0.2)

    print(groot.sorted_nodes[0].loss)
    plt.pause(0.1)

plt.show()
