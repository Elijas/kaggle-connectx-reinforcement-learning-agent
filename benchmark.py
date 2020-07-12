#!/usr/bin/env python
import functools
import multiprocessing as mp
import os
import pkgutil
from collections import Counter
from multiprocessing import Pool

import click
import contexttimer
import kaggle_environments

import agents


def _get_all_agents():
    return [(module_name, module.find_module(module_name).load_module(module_name).act)
            for module, module_name, is_package
            in pkgutil.iter_modules([x[0] for x in os.walk(agents.__path__[0])])
            if not is_package]


def _get_all_enemies(agent_name):
    return [a for a in _get_all_agents() if a[0] not in (agent_name, 'crash', 'human')]


def _parse_results(results):
    """
    >>> _parse_results([[1, -1], [-1, 1], [0, 0], [1, -1]])
    (2, 1)
    """
    counter = Counter(tuple(k) for k in results)
    # Checking against crashes. 'None' value instead of a number means a crashed agent.
    for k in counter.keys():
        assert k in ((1, -1), (0, 0), (-1, 1)), k

    wins = counter.get((1, -1), 0)
    draws = counter.get((0, 0), 0)
    return wins, draws


def _get_results(agent, enemy, game_count):
    assert game_count >= 1
    connectx = functools.partial(kaggle_environments.evaluate, "connectx")

    results = connectx([agent, enemy], num_episodes=game_count - game_count // 2)
    if game_count >= 2:
        results += [reversed(k) for k in connectx([enemy, agent], num_episodes=game_count // 2)]
    return _parse_results(results)


def _benchmark_1v1(agent_name, agent_fun, enemy_name, enemy_fun, game_count):
    wins, draws = _get_results(agent_fun, enemy_fun, game_count)
    print(f"{100 * wins / game_count:6.2f}% "
          f"({wins}/{draws}/{game_count - wins - draws}) [\"{agent_name}\" vs \"{enemy_name}\"]")


@click.command()
@click.argument('agent_name')
@click.option('--enemy-name', '-e', default='__ALL__',
              help='Name of the opposing agent. If not set, plays against all other agents.')
@click.option('--game-count', '-n', default=1, type=click.IntRange(min=1), help='Count of games to be played.')
def benchmark(agent_name, enemy_name, game_count):
    with contexttimer.Timer() as t:
        agent_fun = agents.get_fun(agent_name)

        if enemy_name == '__ALL__':
            enemies = _get_all_enemies(agent_name)
        else:
            enemies = [(enemy_name, agents.get_fun(enemy_name))]

        pool = Pool(mp.cpu_count())
        for enemy_name, enemy_fun in enemies:
            pool.apply_async(_benchmark_1v1, (agent_name, agent_fun, enemy_name, enemy_fun, game_count))
        pool.close()
        pool.join()

        print(f"Completed in {t.elapsed:.1f}s")


if __name__ == '__main__':
    benchmark()
