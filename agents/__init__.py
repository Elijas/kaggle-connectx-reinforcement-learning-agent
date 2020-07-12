import importlib

from agents import simple
from agents.public import tomo20180402


def get_fun(agent_name: str):
    return importlib.import_module(f"agents.{agent_name.replace('/', '.')}").act
