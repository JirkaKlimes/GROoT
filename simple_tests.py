import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.losses import MSE
import keras.backend as K
import os


# model = Sequential()
# model.add(Input((1,)))
# model.add(Dense(32, 'relu'))
# model.add(Dense(32, 'relu'))
# model.add(Dense(1, activation='tanh'))

# x_train = np.linspace(-np.pi, np.pi, 128)
# y_train = np.sin(x_train)


model = Sequential()
model.add(Input((2,)))
model.add(Dense(4, 'relu'))
model.add(Dense(4, 'relu'))
model.add(Dense(2, activation='sigmoid'))

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])

# model.compile(loss='mse')
# model.fit(x_train, y_train, 32, 1000)
# quit()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    from src.groot import GROoT
    from src.cache import PythonDictCache
    from src.planter import Planter

    model.summary()

    dims = Planter.get_dims(model)

    groot = GROoT(dims, 128, cache=PythonDictCache(
        16384), initial_loc=10, initial_scale=5, use_cuda_sampling=True)
    fig = plt.figure()
    loss_kde_ax = fig.add_subplot(1, 2, 1)
    loss_ax = fig.add_subplot(1, 2, 2)

    while True:
        nodes = groot.create_nodes()

        planter = Planter(model, [node.get_position() for node in nodes])
        planter.compile(loss='mse')
        yy = planter(x_train)
        for node, y in zip(nodes, yy):
            loss = np.abs(y - y_train).mean()
            node.loss = loss

        groot.add_rated_nodes(nodes)

        loss_kde_ax.cla()
        losses = list(map(lambda n: n.loss, groot.sorted_nodes))
        kde = gaussian_kde(losses)
        x = np.linspace(max(losses), min(losses), 100)
        y = kde(x)
        loss_kde_ax.plot(x, y)
        loss_kde_ax.set_yscale('log')

        loss_ax.set_yscale('log')
        loss_ax.plot(len(groot.sorted_nodes), groot.sorted_nodes[0].loss, 'g.')

        os.system('clear')
        print(groot)
        plt.pause(0.001)

    # # Compile the model
    # model.compile(optimizer='adam', loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    # # Train the model
    # model.fit(x_train, y_train, validation_data=(
    #     x_test, y_test), epochs=5, batch_size=64)

    # # Evaluate the model on the test set
    # loss, accuracy = model.evaluate(x_test, y_test)
    # print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')
