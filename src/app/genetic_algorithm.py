import random

from chrome_trex import DinoGame
from genetic_dino_agent import GeneticDinoAgent


def evaluate_agent(agent, game):
    game.reset()
    total_score = 0
    while not game.game_over:
        game_state = game.get_state()
        game_state = [float(x) for x in game_state]

        action = agent.get_action(game_state)
        game.step(int(action))
        total_score = game.get_score()

    return total_score


def genetic_algorithm(
    generations=30, population_size=30, mutation_rate=0.05, elitism_size=5
):
    game = DinoGame(fps=0)  # FPS ajustado para garantir estabilidade

    population = [GeneticDinoAgent() for _ in range(population_size)]

    for generation in range(generations):
        scores = [(evaluate_agent(agent, game), agent) for agent in population]
        scores.sort(key=lambda x: x[0], reverse=True)

        print(f"Generation {generation + 1}: Best Score = {scores[0][0]}")

        best_agents = [agent for _, agent in scores[:elitism_size]]
        new_population = best_agents.copy()

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(best_agents, 2)
            child = GeneticDinoAgent.crossover(parent1, parent2)
            child.mutate(mutation_rate)
            new_population.append(child)

        population = new_population

    best_agent = max(population, key=lambda agent: evaluate_agent(agent, game))
    print(f"Best agent's final score: {evaluate_agent(best_agent, game)}")
    game.close()


if __name__ == "__main__":
    generations = 50  # Aumenta o número de gerações para mais aprendizado
    population_size = 30
    mutation_rate = 0.05  # Maior taxa de mutação
    elitism_size = 5

    genetic_algorithm(generations, population_size, mutation_rate, elitism_size)
