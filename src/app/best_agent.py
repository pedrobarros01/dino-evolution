import json
import numpy as np
from dinogame import DinoGame
from NeuralNetworkAgent import NeuralNetworkAgent




game = DinoGame(fps=0)
INPUT_SIZE = 10  
HIDDEN_SIZE = 5  

melhor_agente = NeuralNetworkAgent(INPUT_SIZE, HIDDEN_SIZE)

with open("src\\agents\\best_agent.json", "r") as f:
    weights = json.load(f)
    melhor_agente.weights_input_hidden = np.array(weights["weights_input_hidden"])
    melhor_agente.weights_hidden_output = np.array(weights["weights_hidden_output"])

while not game.game_over:
    game_state = game.get_state()
    action = melhor_agente.get_action(game_state)
    game.step(action)
    scores = game.get_score()
    

scores = game.get_scores()
print(f"{scores=}")
