#!/usr/bin/env python
import click
import kaggle_environments
from IPython import get_ipython
from IPython.core.display import clear_output

import agents

try:
    IS_JUPYTER_ENV = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
except NameError:
    IS_JUPYTER_ENV = False


def _show_board(env):
    if IS_JUPYTER_ENV:
        clear_output(wait=True)
        env.render(mode="ipython", width=500, height=450)
    else:
        print(env.render(mode='ansi'))


def _get_agent_fun(agent_name, env):
    agent_fun = agents.get_fun(agent_name)

    if agent_name == 'human':
        def human_agent_fun(observation, configuration):
            _show_board(env)
            return agent_fun(observation, configuration)

        return human_agent_fun

    return agent_fun


def play(agent_a_name, agent_b_name):
    env = kaggle_environments.make('connectx', debug=True)
    agent_a_fun = _get_agent_fun(agent_a_name, env)
    agent_b_fun = _get_agent_fun(agent_b_name, env)

    trainer = env.train([None, agent_b_fun])  # or: [agent_a_fun, None]
    observation = trainer.reset()
    while not env.done:
        action = agent_a_fun(observation, env.configuration)  # or: agent_b_fun
        observation, reward, done, info = trainer.step(action)

    _show_board(env)
    assert env.state[0].reward is not None, f'Agent A "{agent_a_name}" crashed'
    assert env.state[1].reward is not None, f'Agent B "{agent_b_name}" crashed'
    if env.state[0].reward > env.state[1].reward:
        print(f'Agent A "{agent_a_name}" won the game.')
    elif env.state[0].reward < env.state[1].reward:
        print(f'Agent B "{agent_b_name}" won the game.')
    else:
        print('Game ended in a draw.')


@click.command()
@click.argument('agent_a_name')
@click.argument('agent_b_name')
def play_command(agent_a_name, agent_b_name):
    play(agent_a_name, agent_b_name)


if __name__ == '__main__':
    play_command()
