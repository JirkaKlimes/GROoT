from src.groot import GROoT


if __name__ == '__main__':
    import numpy as np

    DIMS = 2
    BRANCH_FACTOR = 16

    groot = GROoT(DIMS, BRANCH_FACTOR)

    for _ in range(10):
        nodes = groot.create_nodes()
        for node in nodes:
            node.loss = np.random.random()

        groot.add_rated_nodes(nodes)

    for node in groot.sorted_nodes:
        print(node.loss)
