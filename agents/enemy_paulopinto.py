def act(observation, configuration):
    from kaggle_environments.envs.connectx import connectx as ctx
    if sum(observation.board) < configuration.inarow+3:
        return configuration.columns//2
    else:
        return ctx.negamax_agent(observation, configuration)
