import importlib

from agents import simple
from agents.public import tomo20180402
from lab.muzero.agent import act as muzero


def get_fun(agent_name: str):
    if agent_name == 'muzero':
        return muzero

    return importlib.import_module(f"agents.{agent_name.replace('/', '.')}").act
