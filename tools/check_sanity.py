"""
Answers a question: does the game not crash with the given agent when it's playing a match against itself?
"""
import importlib
import sys

import kaggle_environments


def run_sanity_check(agent):
    evaluation = kaggle_environments.evaluate("connectx", [agent, agent], num_episodes=1)
    assert all(r[0] is not None and r[1] is not None for r in evaluation), 'Agent has crashed'


if __name__ == '__main__':
    agent_name = sys.argv[1]
    agent_module = importlib.import_module(f'agents.{agent_name}')
    run_sanity_check(agent_module.act)
