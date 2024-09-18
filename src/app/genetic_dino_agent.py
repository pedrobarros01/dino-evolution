import random
import numpy as np

class GeneticDinoAgent:
    def __init__(self, weights=None, activation_function="relu", bias=None):
        if weights is None:
            self.weights = self.random_weights()
        else:
            self.weights = weights

        if bias is None:
            self.bias = self.random_bias()
        else:
            self.bias = bias

        self.activation_function = activation_function

    def random_bias(self):
        # Biases para duas camadas: [bias para a camada oculta, bias para a camada de saída]
        return [np.random.uniform(-1, 1, 4).tolist(), np.random.uniform(-1, 1, 2).tolist()]

    def random_weights(self):
        # Pesos para duas camadas: [pesos da entrada para camada oculta, pesos da camada oculta para a saída]
        return [np.random.uniform(-1, 1, (5, 4)).tolist(), np.random.uniform(-1, 1, (4, 2)).tolist()]

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
        # Camada 1: Entrada -> Camada oculta
        Z1 = np.matmul(game_state, self.weights[0]) + self.bias[0]
        A1 = self.relu(Z1)

        # Camada 2: Camada oculta -> Saída
        Z2 = np.matmul(A1, self.weights[1]) + self.bias[1]
        A2 = self.softmax(Z2)

        lista = A2.tolist()
        return lista.index(max(lista))

    def mutate(self, mutation_rate):
        for layer in range(len(self.weights)):
            for i in range(len(self.weights[layer])):
                for j in range(len(self.weights[layer][i])):
                    if random.random() < mutation_rate:
                        self.weights[layer][i][j] += np.random.uniform(-1, 1)

    @classmethod
    def crossover_uniforme(cls, parent1, parent2):
        new_weights = [
            np.array([
                [random.choice([w1, w2]) for w1, w2 in zip(row1, row2)]
                for row1, row2 in zip(parent1.weights[layer], parent2.weights[layer])
            ])
            for layer in range(len(parent1.weights))
        ]
        new_bias = [
            [random.choice([b1, b2]) for b1, b2 in zip(parent1.bias[layer], parent2.bias[layer])]
            for layer in range(len(parent1.bias))
        ]
        return cls(new_weights, parent1.activation_function, new_bias)

    @classmethod
    def crossover_unico(cls, parent1, parent2):
        point = random.randint(0, len(parent1.weights[0]) - 1)
        new_weights = [
            np.array(
                [
                    row1 if i < point else row2
                    for i, (row1, row2) in enumerate(zip(parent1.weights[layer], parent2.weights[layer]))
                ]
            )
            for layer in range(len(parent1.weights))
        ]
        new_bias = [
            parent1.bias[layer][:point] + parent2.bias[layer][point:]
            for layer in range(len(parent1.bias))
        ]
        return cls(new_weights, parent1.activation_function, new_bias)

    @classmethod
    def crossover_blend(cls, parent1, parent2, alpha=0.5):
        new_weights = [
            np.array(
                [
                    [(alpha * w1 + (1 - alpha) * w2) for w1, w2 in zip(row1, row2)]
                    for row1, row2 in zip(parent1.weights[layer], parent2.weights[layer])
                ]
            )
            for layer in range(len(parent1.weights))
        ]
        new_bias = [
            [(alpha * b1 + (1 - alpha) * b2) for b1, b2 in zip(parent1.bias[layer], parent2.bias[layer])]
            for layer in range(len(parent1.bias))
        ]
        return cls(new_weights, parent1.activation_function, new_bias)

    def clone(self):
        return GeneticDinoAgent(self.weights, self.activation_function, self.bias)
