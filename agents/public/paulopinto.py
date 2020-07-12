def act(observation, configuration):
    assert configuration.columns == 7 and configuration.rows == 6 and configuration.inarow == 4

    from kaggle_environments.envs.connectx import connectx as ctx
    if sum(observation.board) < configuration.inarow + 3:
        return configuration.columns // 2
    else:
        return ctx.negamax_agent(observation, configuration)
