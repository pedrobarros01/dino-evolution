import random

from chrome_trex import DinoGame, MultiDinoGame
from genetic_dino_agent import GeneticDinoAgent


def process_game_state(game_state):
    """
    Process the game state into a simplified format.

    There can be up to 02 Cacti and 01 Ptera on the screen at a time.
    This function returns a list of states with 11 values for each dino:
    [DY, X_C1, Y_C1, H_C1, X_C2, Y_C2, H_C2, X_P1, Y_P1, H_P1, GS]

    DY is the dinosaur's position in the Y axis;
    X1, Y1, H1 is the first obstacle's X, Y positions and height;
    GS is the game speed.
    """
    DY, H, W, GS, D= game_state

    return [DY, H, W, GS, D]


def reward_function(current_score, previous_score, state):
    # Reward based on score increase and distance traveled
    DY, X1, GS, Y3, X3 = state
    score_increase = current_score - previous_score
    distance = X1
    if X1 > X3:
        distance = X3
    return score_increase + GS * distance


def evaluate_agent(agents, game):
    game.reset()

    while not game.game_over:
        game_state = game.get_state_2()
        actions = [
            agent.get_action(process_game_state(game_state[i]))
            for i, agent in enumerate(agents)
        ]
        game.step(actions)

        # Update distance traveled and score
        scores = game.get_scores()


    agent_with_score = [
        (score, agent)
        for score, agent in zip(scores, agents)
    ]

    return agent_with_score

def tournament_selection(agents_with_scores, tournament_size=3):
    """
    Seleção por torneio. Seleciona um grupo aleatório de agentes (tamanho definido pelo parâmetro `tournament_size`)
    e retorna o agente com a maior pontuação no grupo.
    """
    tournament_contestants = random.sample(agents_with_scores, tournament_size)
    best_agent = max(tournament_contestants, key=lambda x: x[0])  # Seleciona o agente com maior score
    return best_agent[1]  # Retorna o agente


def roulette_wheel_selection(agents_with_scores):
    """
    Seleção por roleta. A probabilidade de um agente ser selecionado
    é proporcional à sua pontuação.
    """
    total_score = sum(score for score, _ in agents_with_scores)
    selection_probs = [score / total_score for score, _ in agents_with_scores]
    
    return random.choices(
        [agent for _, agent in agents_with_scores], 
        weights=selection_probs, 
        k=1
    )[0]

def genetic_algorithm(
    generations=1, population_size=30, mutation_rate=0.05, elitism_size=2
):
    game = MultiDinoGame(fps=0, dino_count=population_size)

    population = [
        GeneticDinoAgent(activation_function="sigmoid")
        for _ in range(population_size)
    ]

    best_global_agent = None
    best_global_score = float("-inf")
    best_generation_agent = None
    best_generation_score = float("-inf")

    for generation in range(generations):
        scores = evaluate_agent(population, game)
        scores.sort(key=lambda x: x[0], reverse=True)

        # Capture the best agent in this generation
        best_generation_score = scores[0][0]
        best_generation_agent = scores[0][1]

        # Update the global best agent if necessary
        if best_generation_score > best_global_score:
            best_global_score = best_generation_score
            best_global_agent = best_generation_agent.clone()

        print(f"Generation {generation + 1}: Best Score = {best_generation_score}")

        # Selection for the next generation
        best_agents = [agent for _, agent in scores[:elitism_size]]
        new_population = best_agents.copy()
        new_population.append(best_global_agent)

        while len(new_population) < population_size:
            parent1 = roulette_wheel_selection(scores)
            parent2 = roulette_wheel_selection(scores)
            child = GeneticDinoAgent.crossover_unico(parent1, parent2)
            child.mutate(mutation_rate)
            new_population.append(child)

        population = new_population

    print(f"Best global agent's final score: {best_global_score}")
    game.close()

    return best_global_agent


def evaluate_agent_performance(agent, num_trials=10):
    total_scores = []

    for _ in range(num_trials):
        game = DinoGame(fps=0)
        game.reset()

        while not game.game_over:
            game_state = game.get_state()
            processed_state = process_game_state(game_state)
            action = agent.get_action(processed_state)
            game.step(action)

        scores = game.get_score()
        game.close()
        total_scores.append(scores)

    average_score = sum(total_scores) / len(total_scores)
    print(f"Pontuação média do melhor agente global: {average_score}")
    return average_score


if __name__ == "__main__":
    generations = 50  # Increase the number of generations for more learning
    population_size = 50
    mutation_rate = 1  # Higher mutation rate
    elitism_size = 4

    best_agent = genetic_algorithm(
        generations, population_size, mutation_rate, elitism_size
    )
    evaluate_agent_performance(best_agent)
