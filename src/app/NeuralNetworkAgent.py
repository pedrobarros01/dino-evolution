import random

import numpy as np
from dinogame import ACTION_DOWN, ACTION_UP

# Definição do agente com uma rede neural simples
class NeuralNetworkAgent:
    def __init__(self, input_size, hidden_size):
        # Inicializando os pesos da rede neural (1 camada oculta)
        self.weights_input_hidden = np.random.uniform(
            -1000, 1000, (input_size, hidden_size)
        )
        self.weights_hidden_output = np.random.uniform(-1000, 1000, hidden_size)

    def relu(self, x):
        return np.maximum(0, x)

    def get_action(self, state):
        # Forward pass: entrada -> camada oculta -> saída
        hidden_layer = self.relu(np.dot(state, self.weights_input_hidden))
        output = self.relu(np.dot(hidden_layer, self.weights_hidden_output))

        # Apenas duas ações possíveis: ACTION_UP ou ACTION_DOWN
        if output > 0:
            return ACTION_UP
        else:
            return ACTION_DOWN

    def mutate(self, mutation_rate=0.05, mutation_intensity=1000):
        # Mutar os pesos da camada input -> hidden
        for i in range(self.weights_input_hidden.shape[0]):
            for j in range(self.weights_input_hidden.shape[1]):
                if random.random() < mutation_rate:
                    self.weights_input_hidden[i, j] = random.uniform(
                        -mutation_intensity, mutation_intensity
                    )

        # Mutar os pesos da camada hidden -> output
        for i in range(len(self.weights_hidden_output)):
            if random.random() < mutation_rate:
                self.weights_hidden_output[i] = random.uniform(
                    -mutation_intensity, mutation_intensity
                )