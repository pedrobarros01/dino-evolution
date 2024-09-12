import random
from chrome_trex import DinoGame
from genetic_dino_agent import GeneticDinoAgent

def evaluate_agent(agent, game):
    # Reinicia o jogo para cada agente
    game.reset()
    total_score = 0
    while not game.game_over:
        # Obtém o estado do jogo
        game_state = game.get_state()        
        game_state = [float(x) for x in game_state]
        
        # O agente decide qual ação tomar com base no estado
        action = agent.get_action(game_state)
        print(f"Action: {action}, Type: {type(action)}")
        
        # O jogo avança com a ação selecionada
        game.step(int(action)) 
        # Obtém a pontuação atual do jogo
        total_score = game.get_score()
    
    return total_score

def genetic_algorithm(generations=10, population_size=20, mutation_rate=0.01):
    # Cria o jogo com FPS ilimitado
    game = DinoGame(fps=60)  # Ajustei o FPS para evitar problemas com frames

    # Cria uma população inicial de agentes
    population = [GeneticDinoAgent() for _ in range(population_size)]
    
    for generation in range(generations):
        # Avalia a pontuação de cada agente
        scores = [(evaluate_agent(agent, game), agent) for agent in population]
        scores.sort(key=lambda x: x[0], reverse=True)  # Ordena os agentes pelo score

        print(f"Generation {generation + 1}: Best Score = {scores[0][0]}")

        # Seleciona os 5 melhores agentes para reprodução
        best_agents = [agent for _, agent in scores[:5]]
        new_population = []
        
        # Gera a nova população através de cruzamento e mutação
        for _ in range(population_size):
            parent1, parent2 = random.sample(best_agents, 2)
            child = GeneticDinoAgent.crossover(parent1, parent2)
            child.mutate(mutation_rate)
            new_population.append(child)

        population = new_population

    # Avalia o melhor agente ao final
    best_agent = max(population, key=lambda agent: evaluate_agent(agent, game))
    print(f"Best agent's final score: {evaluate_agent(best_agent, game)}")
