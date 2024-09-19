import random
import numpy as np
from dinogame import ACTION_DOWN, ACTION_FORWARD, ACTION_UP, MultiDinoGame, DinoGame
import random

# Definição do agente com uma rede neural simples
class NeuralNetworkAgent:
    def __init__(self, input_size, hidden_size):
        # Inicializando os pesos da rede neural (1 camada oculta)
        self.weights_input_hidden = np.random.uniform(-1000, 1000, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1000, 1000, hidden_size)

    def relu(self, x):
        return np.maximum(0, x)

    def get_action(self, state):
        # Forward pass: entrada -> camada oculta -> saída
        hidden_layer = self.relu(np.dot(state, self.weights_input_hidden))
        output = self.relu(np.dot(hidden_layer, self.weights_hidden_output))
        
        # Apenas duas ações possíveis: ACTION_UP ou ACTION_DOWN
        if output > 0:
            return ACTION_UP
        else:
            return ACTION_DOWN

    def mutate(self, mutation_rate=0.05, mutation_intensity=1000):
        # Mutar os pesos da camada input -> hidden
        for i in range(self.weights_input_hidden.shape[0]):
            for j in range(self.weights_input_hidden.shape[1]):
                if random.random() < mutation_rate:
                    self.weights_input_hidden[i, j] = random.uniform(-mutation_intensity, mutation_intensity)
        
        # Mutar os pesos da camada hidden -> output
        for i in range(len(self.weights_hidden_output)):
            if random.random() < mutation_rate:
                self.weights_hidden_output[i] = random.uniform(-mutation_intensity, mutation_intensity)

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
        if best_score > 10000:
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
POPULATION_SIZE = 50
GENERATIONS = 1000
INPUT_SIZE = 10   # Tamanho da entrada da rede neural (estado do jogo)
HIDDEN_SIZE = 5   # Número de neurônios na camada oculta

# Iniciar jogo
'''game = MultiDinoGame(fps=0, dino_count=POPULATION_SIZE)
best_agent = genetic_algorithm(POPULATION_SIZE, GENERATIONS, INPUT_SIZE, HIDDEN_SIZE, game)
file = open('melhor_dino.txt', 'w+')
file.write(f'{best_agent.weights_input_hidden=}\n')
file.write(f'{best_agent.weights_hidden_output=}\n')
file.close()'''
game = DinoGame(fps=240)
melhor_agente = NeuralNetworkAgent(INPUT_SIZE, HIDDEN_SIZE)
melhor_agente.weights_input_hidden =   np.array([[ 800.99227089, -480.48063939, -301.2534357 , -196.46536498,
        -743.09150888],
       [ 960.52639824,  171.58194787, -183.25082742,  -79.47714778,
        -703.96205886],
       [  55.66500212, -904.74403859,  977.0442675 ,  442.07062394,
        -767.35275156],
       [-716.82480082,  -66.64107161,  630.6242964 , -770.78175814,
        -604.12623559],
       [ 132.99169764,  204.99824639,  689.51118639, -998.88609201,
         272.33229074],
       [-577.96391641, -974.58688859,  198.7438941 , -597.20657665,
        -184.1173736 ],
       [ 885.94358581, -566.66038815,  196.79556628, -683.92669424,
         882.00690058],
       [ 224.10414014,   45.33285731,  121.03031629,  -70.23585472,
        -125.29485757],
       [ 722.54682639, -421.63810415, -185.24726454,  701.27940819,
         575.94632705],
       [-128.19590182,  914.84045664,  -98.24447554,  222.78941436,
         296.99801165]])
melhor_agente.weights_hidden_output = np.array([-939.90476056,  -89.59455454,  282.4566539 , -757.53241987,
        552.09734337])
while not game.game_over:
    game_state = game.get_state()
    action = melhor_agente.get_action(game_state)
    game.step(action)

scores = game.get_scores()
print(f'{scores=}')
