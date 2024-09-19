import random
import numpy as np
from dinogame import ACTION_DOWN, ACTION_FORWARD, ACTION_UP, MultiDinoGame, DinoGame
import random
import json

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

game = DinoGame(fps=0)
INPUT_SIZE = 10   # Tamanho da entrada da rede neural (estado do jogo)
HIDDEN_SIZE = 5   # Número de neurônios na camada oculta

melhor_agente = NeuralNetworkAgent(INPUT_SIZE, HIDDEN_SIZE)

with open('../agents/best_agent.json', 'r') as f:
    weights = json.load(f)
    melhor_agente.weights_input_hidden = np.array(weights['weights_input_hidden'])
    melhor_agente.weights_hidden_output = np.array(weights['weights_hidden_output'])

while not game.game_over:
    game_state = game.get_state()
    action = melhor_agente.get_action(game_state)
    game.step(action)

scores = game.get_scores()
print(f'{scores=}')
