import kaggle_environments

from bots import always_select_first

NUM_EPISODES = 500
AGENTS = [
    'random',
    'negamax',
    always_select_first.act
]


def evaluate_1v1(agent1, agent2, num_episodes):
    def mean_reward(rewards):
        return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)

    evaluation = kaggle_environments.evaluate("connectx", [agent1, agent2], num_episodes=num_episodes)
    return mean_reward(evaluation)


def evaluate_vs_others(agent, enemy_agents=None):
    enemy_agents = [a for a in enemy_agents or AGENTS if a != agent]
    assert enemy_agents
    return sum(evaluate_1v1(agent, enemy_agent, NUM_EPISODES) for enemy_agent in enemy_agents) / len(enemy_agents)


if __name__ == '__main__':
    print(evaluate_1v1('negamax', 'random', 100))
    #print(evaluate_vs_others(always_select_first.act))
