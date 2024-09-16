import random

import numpy as np
from chrome_trex import MultiDinoGame

# Definições básicas para o algoritmo genético
population_size = 50  # Tamanho da população
generations = 50  # Número de gerações
mutation_rate = 0.05  # Taxa de mutação

# Definições de ações
ACTION_FORWARD = 0
ACTION_UP = 1
ACTION_DOWN = 2


class Agent:
    def __init__(self):
        self.weights = np.random.rand(11, 3)  # Inicializa pesos aleatórios
        self.fitness = 0

    def get_action(self, state):
        # Converte o estado para um array numpy
        state = np.array(state)

        # Calcula o valor para cada ação com base no estado e pesos
        scores = np.dot(state, self.weights)

        # Escolhe a ação com a maior pontuação
        action = np.argmax(scores)

        return action


def process_game_state(state):
    # Processar o estado do jogo para a entrada do agente
    return state


def evaluate_agents(agents, game):
    for i, agent in enumerate(agents):
        game.reset()
        while not game.game_over:
            game_state = game.get_state()[i]
            processed_state = process_game_state(game_state)
            actions = [agent.get_action(processed_state) for agent in agents]
            game.step(actions)
        agent.fitness = game.get_scores()[i]


def select_parents(agents):
    # Seleção dos pais com base na aptidão
    sorted_agents = sorted(agents, key=lambda a: a.fitness, reverse=True)
    return sorted_agents[:2]  # Selecionar os melhores dois


def crossover(parent1, parent2):
    child = Agent()
    split = random.randint(0, len(parent1.weights) - 1)
    child.weights[:split] = parent1.weights[:split]
    child.weights[split:] = parent2.weights[split:]
    return child


def mutate(agent):
    mutation_mask = np.random.rand(*agent.weights.shape) < mutation_rate
    agent.weights[mutation_mask] += np.random.randn(*agent.weights.shape)[mutation_mask]


def genetic_algorithm():
    game = MultiDinoGame(fps=0, dino_count=population_size)

    agents = [Agent() for _ in range(population_size)]

    for generation in range(generations):
        evaluate_agents(agents, game)

        parents = select_parents(agents)

        new_agents = []
        while len(new_agents) < population_size:
            child = crossover(parents[0], parents[1])
            mutate(child)
            new_agents.append(child)

        agents = new_agents

        # Imprimir a melhor aptidão da geração
        best_agent = max(agents, key=lambda a: a.fitness)
        print(f"Generation {generation}: Best Score = {best_agent.fitness}")


if __name__ == "__main__":
    genetic_algorithm()
