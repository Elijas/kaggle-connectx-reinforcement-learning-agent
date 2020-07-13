def act(observation, configuration):
    return [c for c in range(configuration.columns) if observation.board[c] == 0][0]
