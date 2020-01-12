import kaggle_environments


def debug_steps(step_fun, enemy_agent, enemy_starts=False):
    env = kaggle_environments.make('connectx', debug=True)
    trainer = env.train([enemy_agent, None] if enemy_starts else [None, enemy_agent])
    observation = trainer.reset()
    while not env.done:
        action = step_fun(observation, env)
        observation, reward, done, info = trainer.step(action)

    env.render()
    assert env.state[0].reward is not None, 'Player 1 crashed'
    assert env.state[1].reward is not None, 'Player 2 crashed'
    if env.state[0].reward > env.state[1].reward:
        print('Player 1 won')
    elif env.state[0].reward < env.state[1].reward:
        print('Player 2 won')
    else:
        print('Draw')
