from agents import selects_leftmost
from tools._debug import debug_steps


def step(observation, env):
    action = selects_leftmost.act(observation, env.configuration)
    env.render()
    print("Selected action:", action)
    return action


if __name__ == '__main__':
    debug_steps(step, enemy_agent=selects_leftmost.act)
