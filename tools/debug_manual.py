from agents import selects_leftmost
from tools._debug_steps import debug_steps


def step_manual(observation, env):
    env.render()
    return int(input('Enter action number: '))


if __name__ == '__main__':
    enemy_starts = False
    print(f'Indexing starts at 0. You are Player {2 if enemy_starts else 1}')
    debug_steps(step_manual, enemy_agent=selects_leftmost.act, enemy_starts=enemy_starts)
