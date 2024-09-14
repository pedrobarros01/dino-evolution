import random

from chrome_trex import MultiDinoGame
from genetic_dino_agent import GeneticDinoAgent


def process_game_state(game_state):
    DY, X1, Y1, H1, X2, Y2, H2, X3, Y3, H3, GS = game_state
    return [DY, X1, H1, GS]


def reward_function(current_score, previous_score, distance):
    # A recompensa é baseada no aumento de pontuação e na distância percorrida
    score_increase = current_score - previous_score
    return score_increase + 0.1 * distance


def evaluate_agent(agents, game):
    game.reset()
    total_scores = [0] * len(agents)
    previous_scores = [0] * len(agents)

    while not game.game_over:
        game_state = game.get_state()[0]
        processed_state = process_game_state(game_state)  # Processar o estado

        actions = [agent.get_action(processed_state) for agent in agents]
        game.step(actions)

        # Atualize a distância percorrida e a pontuação
        scores = game.get_scores()
        for i, (score, agent) in enumerate(zip(scores, agents)):
            distance = game_state[1]  # X1 - Distância do obstáculo mais próximo
            total_scores[i] += reward_function(score, previous_scores[i], distance)
            previous_scores[i] = score

    agent_with_score = [(score, agent) for score, agent in zip(total_scores, agents)]
    return agent_with_score


def genetic_algorithm(
    generations=1, population_size=30, mutation_rate=0.05, elitism_size=2
):
    game = MultiDinoGame(fps=0, dino_count=population_size)

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
    generations = 100  # Aumenta o número de gerações para mais aprendizado
    population_size = 50
    mutation_rate = 0.05  # Maior taxa de mutação
    elitism_size = 5

    genetic_algorithm(generations, population_size, mutation_rate, elitism_size)
