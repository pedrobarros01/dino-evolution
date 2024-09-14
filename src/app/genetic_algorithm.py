import random

from chrome_trex import MultiDinoGame
from genetic_dino_agent import GeneticDinoAgent


def evaluate_agent(agents, game):
    game.reset()
    total_score = 0
    while not game.game_over:
        game_state = game.get_state()[0]
        game_state = [float(game_state[x]) for x in [0, 1, 2, 3, 10]]
        actions = []
        for agent in agents:
            action = agent.get_action(game_state)
            actions.append(int(action))
        game.step(actions)
    total_score = game.get_scores()
    agent_with_score = [(score, agent) for score, agent in zip(total_score, agents)]

    return agent_with_score


def genetic_algorithm(
    generations=1, population_size=30, mutation_rate=0.05, elitism_size=2
):
    game = MultiDinoGame(
        fps=0, dino_count=population_size
    )  # FPS ajustado para garantir estabilidade

    population = [
        GeneticDinoAgent(activation_function="leaky_relu")
        for _ in range(population_size)
    ]

    for generation in range(generations):
        scores = evaluate_agent(population, game)
        scores.sort(key=lambda x: x[0], reverse=True)
        # print(scores)
        print(f"Generation {generation + 1}: Best Score = {scores[0][0]}")

        best_agents = [agent for _, agent in scores[:elitism_size]]
        new_population = best_agents.copy()

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(best_agents, 2)
            child = GeneticDinoAgent.crossover(parent1, parent2)
            child.mutate(mutation_rate)
            new_population.append(child)

        population = new_population

    best_agent = max(scores, key=lambda x: x[0])
    print(f"Best agent's final score: {best_agent[0]}")
    game.close()


if __name__ == "__main__":
    generations = 50  # Aumenta o número de gerações para mais aprendizado
    population_size = 30
    mutation_rate = 0.05  # Maior taxa de mutação
    elitism_size = 5

    genetic_algorithm(generations, population_size, mutation_rate, elitism_size)
