import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical


# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create a simple CNN model
model = Sequential()
model.add(Input((28, 28, 1)))
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


def get_weights(model):
    weights = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if layer_weights:
            weights.extend([w.flatten() for w in layer_weights])
    return np.concatenate(weights)


def set_weights(model, weights_vector):
    index = 0
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if layer_weights:
            new_weights = []
            for w in layer_weights:
                shape = w.shape
                size = np.prod(shape)
                new_weights.append(
                    weights_vector[index:index + size].reshape(shape))
                index += size
            layer.set_weights(new_weights)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    from src.groot import GROoT
    from src.cache import PythonDictCache

    model.compile(loss='categorical_crossentropy')
    model.summary()

    dims = len(get_weights(model))

    groot = GROoT(dims, 32, cache=PythonDictCache(1024))
    fig = plt.figure()
    loss_ax = fig.add_subplot()

    while True:
        nodes = groot.create_nodes()

        for n in nodes:
            set_weights(model, n.get_position())
            loss = model.evaluate(x_train[:100], y_train[:100], verbose=1)
            n.loss = loss

        groot.add_rated_nodes(nodes)

        loss_ax.cla()
        losses = list(map(lambda n: n.loss, groot.sorted_nodes))
        kde = gaussian_kde(losses)
        x = np.linspace(max(losses), min(losses), 100)
        y = kde(x)
        loss_ax.plot(x, y)
        loss_ax.set_yscale('log')

        print(
            f"GROoT's loss: {groot.sorted_nodes[0].loss}")
        plt.pause(1)

    # # Compile the model
    # model.compile(optimizer='adam', loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    # # Train the model
    # model.fit(x_train, y_train, validation_data=(
    #     x_test, y_test), epochs=5, batch_size=64)

    # # Evaluate the model on the test set
    # loss, accuracy = model.evaluate(x_test, y_test)
    # print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')
