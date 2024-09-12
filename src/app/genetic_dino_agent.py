import random

import numpy as np
from chrome_trex import ACTION_DOWN, ACTION_FORWARD, ACTION_UP


class GeneticDinoAgent:
    def __init__(self, weights=None):
        # O tamanho dos pesos deve ser o mesmo do vetor de estado (10)
        self.weights = weights if weights is not None else self.random_weights()
        self.bias = random.randint(0, 1)

    def random_weights(self):
        # O vetor de estado tem 10 elementos, então precisamos de 10 pesos
        return np.random.randn(10)

    def get_action(self, game_state):
        # Convertemos o estado do jogo para um array NumPy
        input_data = np.array(game_state)

        # Fazemos o produto escalar entre os pesos e o vetor de estado
        result = np.dot(self.weights, input_data) + self.bias

        # Usamos o resultado para decidir qual ação tomar
        if result > 0.5:
            return ACTION_UP  # Pular
        elif result < -0.5:
            return ACTION_DOWN  # Abaixar
        else:
            return ACTION_FORWARD  # Continuar correndo

    def mutate(self, mutation_rate=0.01):
        # Aplica mutação nos pesos
        for i in range(len(self.weights)):
            if random.random() < mutation_rate:
                self.weights[i] += np.random.randn()

    @classmethod
    def crossover(cls, parent1, parent2):
        # Faz cruzamento dos pesos de dois pais
        new_weights = np.array(
            [(w1 + w2) / 2 for w1, w2 in zip(parent1.weights, parent2.weights)]
        )
        return cls(new_weights)
