import kaggle_environments

from bots import always_select_first

NUM_EPISODES = 100
AGENTS = [
    'random',
    always_select_first.act
]


def evaluate_1v1(agent1, agent2, num_episodes):
    def mean_reward(rewards):
        return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)

    evaluation = kaggle_environments.evaluate("connectx", [agent1, agent2], num_episodes=num_episodes)
    return mean_reward(evaluation)


def evaluate_vs_everyone(agent):
    enemy_agents = [a for a in AGENTS if a != agent]
    assert enemy_agents
    score = sum(evaluate_1v1(agent, enemy_agent, NUM_EPISODES) for enemy_agent in enemy_agents) / len(enemy_agents)
    return score


if __name__ == '__main__':
    agent = always_select_first.act
    print(evaluate_vs_everyone(agent))
