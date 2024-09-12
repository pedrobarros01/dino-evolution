from chrome_trex import ACTION_UP, DinoGame

# Create a new game that runs with at most 'fps' frames per second.
# Use fps=0 for unlimited fps.
game = DinoGame(fps=60)

while not game.game_over:
    # Go to the next frame and take the action 'action'
    # (ACTION_UP, ACTION_FORWARD or ACTION_DOWN).
    game.step(ACTION_UP)

    # Get a list of floats representing the game state
    # (positions of the obstacles and game speed).
    game.get_state()
    print(game.get_state())

    # Get the game score.
    game.get_score()
