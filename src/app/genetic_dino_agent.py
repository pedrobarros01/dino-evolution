import random

import numpy as np
from chrome_trex import ACTION_DOWN, ACTION_FORWARD, ACTION_UP


class GeneticDinoAgent:
    def __init__(self, weights=None, activation_function="relu"):
        # O tamanho dos pesos deve ser o mesmo do vetor de estado (10x3)
        self.weights = weights if weights is not None else self.random_weights()
        self.bias = np.random.uniform(-1, 1, 3).tolist()
        self.activation_function = activation_function
        # print(self.bias)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, x * alpha)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)

    def elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def random_weights(self):
        # O vetor de estado tem 10 elementos, então precisamos de 10 pesos
        pesos = []
        for i in range(5):
            # print(i)
            peso_aleatorio = np.random.uniform(
                -1, 1, 3
            ).tolist()  # Inicialização uniforme entre -1 e 1
            pesos.append(peso_aleatorio)
        return pesos

    def activate(self, Z):
        if self.activation_function == "relu":
            return self.relu(Z)
        elif self.activation_function == "sigmoid":
            return self.sigmoid(Z)
        elif self.activation_function == "tanh":
            return self.tanh(Z)
        elif self.activation_function == "leaky_relu":
            return self.leaky_relu(Z)
        elif self.activation_function == "softmax":
            return self.softmax(Z)
        elif self.activation_function == "elu":
            return self.elu(Z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_function}")

    def get_action(self, game_state):
        Z = np.matmul(game_state, self.weights) + self.bias
        Z = self.activate(Z)

        if Z[0] > Z[1] and Z[0] > Z[2]:
            return ACTION_FORWARD
        elif Z[1] > Z[2]:
            return ACTION_UP
        else:
            return ACTION_DOWN

    def mutate(self, mutation_rate=0.05):  # Aumenta a taxa de mutação
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                if random.random() < mutation_rate:
                    self.weights[i][j] += np.random.uniform(
                        -1, 1
                    )  # Mutação mais controlada com valores pequenos

    @classmethod
    def crossover(cls, parent1, parent2):
        new_weights = np.array(
            [
                [
                    random.choice([w1, w2]) for w1, w2 in zip(row1, row2)
                ]  # Crossover uniforme
                for row1, row2 in zip(parent1.weights, parent2.weights)
            ]
        )
        return cls(new_weights)
