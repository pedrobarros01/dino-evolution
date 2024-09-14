import random

import numpy as np
from chrome_trex import ACTION_DOWN, ACTION_FORWARD, ACTION_UP


class GeneticDinoAgent:
    def __init__(self, weights=None):
        # O tamanho dos pesos deve ser o mesmo do vetor de estado (10x3)
        self.weights = weights if weights is not None else self.random_weights()
        self.bias = np.random.uniform(0, 1, 3).tolist()
        # print(self.bias)

    def random_weights(self):
        # O vetor de estado tem 10 elementos, então precisamos de 10 pesos
        pesos = []
        for i in range(11):
            # print(i)
            peso_aleatorio = np.random.uniform(
                0, 1, 3
            ).tolist()  # Inicialização uniforme entre -1 e 1
            pesos.append(peso_aleatorio)
        return pesos

    def relu(self, x):
        return np.maximum(0, x)

    def get_action(self, game_state):
        Z = np.matmul(game_state, self.weights) + self.bias
        Z = self.relu(Z)  # Substitui sigmoid por ReLU

        # Mantém a lógica de decisão de ação
        if Z[0] > Z[1] and Z[0] > Z[2]:
            return ACTION_UP
        elif Z[1] > Z[2]:
            return ACTION_FORWARD
        else:
            return ACTION_DOWN

    def mutate(self, mutation_rate=0.05):  # Aumenta a taxa de mutação
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                if random.random() < mutation_rate:
                    self.weights[i][j] += np.random.uniform(
                        -0.5, 0.5
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
