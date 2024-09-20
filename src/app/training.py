from dinogame import  MultiDinoGame
import json
from NeuralNetworkAgent import NeuralNetworkAgent



def evaluate_agent(agents, game):
    game.reset()
    vivos = [True for _ in agents]
    while not game.game_over:
        game_state = game.get_state()
        actions = [
            agent.get_action(game_state[i])
            for i, agent in enumerate(agents)
        ]
        game.step(actions)
        if max(game.get_scores()) > 30000:
            game.game_over = True 
    scores = game.get_scores()
    agent_with_score = [(score, agent) for score, agent in zip(scores, agents)]
    return agent_with_score

def genetic_algorithm(population_size, generations, input_size, hidden_size, game):
    population = [NeuralNetworkAgent(input_size, hidden_size) for _ in range(population_size)]
    
    for generation in range(generations):
        # Avaliar agentes
        agents_with_scores = evaluate_agent(population, game)
        agents_with_scores.sort(reverse=True, key=lambda x: x[0])  # Ordena pela pontuação
        
        # Selecionar o melhor agente
        best_agent = agents_with_scores[0][1]
        best_score = agents_with_scores[0][0]
        if best_score > 30000:
            print(f"Geração {generation}: Melhor pontuação = {best_score}")
            return best_agent
        # Clonar o melhor agente sem mutação
        new_population = [best_agent]  # Melhor agente direto na nova geração
        
        # Gerar nova população mutando o resto da população
        while len(new_population) < population_size:
            agent_copy = NeuralNetworkAgent(input_size, hidden_size)  # Novo agente
            agent_copy.weights_input_hidden = best_agent.weights_input_hidden.copy()  # Copiar pesos
            agent_copy.weights_hidden_output = best_agent.weights_hidden_output.copy()  # Copiar pesos
            agent_copy.mutate(mutation_rate=0.4)  # Aplica mutação leve
            new_population.append(agent_copy)

        # Nova população inclui o melhor agente clonado + agentes mutados
        population = new_population

        # Exibir progresso
        print(f"Geração {generation}: Melhor pontuação = {best_score}")

    # Retornar o melhor agente
    return max(population, key=lambda agent: evaluate_agent([agent], game)[0][0])

# Parâmetros do algoritmo genético
POPULATION_SIZE = 100
GENERATIONS = 1000
INPUT_SIZE = 10   # Tamanho da entrada da rede neural (estado do jogo)
HIDDEN_SIZE = 5   # Número de neurônios na camada oculta

# Iniciar jogo
game = MultiDinoGame(fps=0, dino_count=POPULATION_SIZE)
best_agent = genetic_algorithm(POPULATION_SIZE, GENERATIONS, INPUT_SIZE, HIDDEN_SIZE, game)

best_agent_dict = {
    'weights_input_hidden': best_agent.weights_input_hidden.tolist(),
    'weights_hidden_output': best_agent.weights_hidden_output.tolist()
}

# save best agent on best_agent.json inside agents folder
with open('src\\agents\\best_agent.json', 'w+') as f:
    json.dump(best_agent_dict, f)