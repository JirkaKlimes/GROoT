import numpy as np
from keras.layers import Input, Concatenate
from keras.models import Model
from keras.models import clone_model
from keras import backend as K


class Planter(Model):

    @staticmethod
    def get_dims(model: Model) -> int:
        return sum([np.prod(w.shape) for w in model.get_weights()])

    def __init__(self, model: Model, weights: list):
        # Step 1: Create a shared input layer for the models
        shared_input = Input(model.input_shape[1:])

        # Step 2: Create a list of cloned models based on the input model
        models = [clone_model(model, shared_input)
                  for _ in range(len(weights))]

        # Step 3: Iterate over each cloned model and update its weights
        for i, model in enumerate(models):
            weights_start = 0
            for layer in model.layers:
                # Update the name of the layers to include the model index
                layer._name = f'{i}_{layer.name}'
                new_weights = []
                for layer_weights in layer.get_weights():
                    # Calculate the length of weights for the current layer
                    length = np.prod(layer_weights.shape)
                    # Extract the corresponding weights from the weights list
                    weight_chunk = weights[i][weights_start:weights_start+length]
                    weights_start += length
                    # Reshape the weights and store them in a new_weights list
                    weight_chunk = np.reshape(
                        weight_chunk, layer_weights.shape)
                    new_weights.append(weight_chunk)
                # Set the updated weights for the layer
                layer.set_weights(new_weights)

        # Step 4: Create a list of model outputs and concatenate them
        outputs = [model.output for model in models]
        outputs = [K.expand_dims(output, 0) for output in outputs]
        outputs = Concatenate(0)(outputs)

        # Step 5: Call the constructor of the parent class with shared_input as input and concatenated outputs as output
        super().__init__(inputs=shared_input, outputs=outputs)
