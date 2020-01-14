import kaggle_environments

from agents import selects_leftmost, enemy_bluefool, enemy_tomo20180402
from tools.patch_with_timer import patch_with_timer

NUM_EPISODES = 10
AGENTS = [
    'random',
    'negamax',
    selects_leftmost.act,
    enemy_bluefool.act,
    enemy_tomo20180402.act
]


def evaluate_1v1(agent1, agent2, num_episodes):
    def mean_reward_agent1(games_results):
        assert all(r[0] is not None for r in games_results), 'Agent1 has crashed'
        assert all(r[1] is not None for r in games_results), 'Agent2 has crashed'
        return sum(r[0] for r in games_results) / sum(r[0] + r[1] for r in games_results)

    games_results = kaggle_environments.evaluate("connectx", [agent1, agent2], num_episodes=num_episodes)
    return mean_reward_agent1(games_results)


def evaluate_vs_others(agent, enemy_agents=None):
    enemy_agents = [a for a in enemy_agents or AGENTS if a != agent]
    assert enemy_agents
    return sum(evaluate_1v1(agent, enemy_agent, NUM_EPISODES) for enemy_agent in enemy_agents) / len(enemy_agents)


if __name__ == '__main__':
    t = patch_with_timer('') # insert the path into the ()
    print(evaluate_1v1(enemy_bluefool.act, 'negamax', NUM_EPISODES))
    print(t)
    # print(evaluate_vs_others(enemy_bluefool.act))
