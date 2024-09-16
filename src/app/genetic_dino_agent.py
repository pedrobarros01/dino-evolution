import random

import numpy as np


class GeneticDinoAgent:
    def __init__(self, weights=None, activation_function="relu", bias=None):
        self.weights = weights if weights is not None else self.random_weights()
        self.bias = bias if bias is not None else self.random_bias()
        self.activation_function = activation_function
        # print(self.bias)

    def random_bias(self):
        return np.random.uniform(-1, 1, 3).tolist()

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
        pesos = []
        for i in range(5):
            # print(i)
            peso_aleatorio = np.random.uniform(-1, 1, 3).tolist()
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
        lista = Z.tolist()
        return lista.index(max(lista))

    def mutate(self, mutation_rate=0.05):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                if random.random() < mutation_rate:
                    self.weights[i][j] += np.random.uniform(-1, 1)

        for i in range(len(self.bias)):
            if random.random() < mutation_rate:
                self.bias[i] += np.random.uniform(-1, 1)

    @classmethod
    def crossover_uniforme(cls, parent1, parent2):
        new_weights = np.array(
            [
                [
                    random.choice([w1, w2]) for w1, w2 in zip(row1, row2)
                ]  # Crossover uniforme
                for row1, row2 in zip(parent1.weights, parent2.weights)
            ]
        )
        new_bias = [
            random.choice([b1, b2]) for b1, b2 in zip(parent1.bias, parent2.bias)
        ]
        return cls(new_weights, parent1.activation_function, new_bias)

    @classmethod
    def crossover_unico(cls, parent1, parent2):
        point = random.randint(0, len(parent1.weights) - 1)
        new_weights = np.array(
            [
                row1 if i < point else row2
                for i, (row1, row2) in enumerate(zip(parent1.weights, parent2.weights))
            ]
        )
        new_bias = parent1.bias[:point] + parent2.bias[point:]
        return cls(new_weights, parent1.activation_function, new_bias)

    @classmethod
    def crossover_blend(cls, parent1, parent2, alpha=0.5):
        new_weights = np.array(
            [
                [(alpha * w1 + (1 - alpha) * w2) for w1, w2 in zip(row1, row2)]
                for row1, row2 in zip(parent1.weights, parent2.weights)
            ]
        )
        new_bias = [
            (alpha * b1 + (1 - alpha) * b2)
            for b1, b2 in zip(parent1.bias, parent2.bias)
        ]
        return cls(new_weights, parent1.activation_function, new_bias)

    def clone(self):
        return GeneticDinoAgent(self.weights, self.activation_function, self.bias)
