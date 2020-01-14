import time
from unittest.mock import Mock
import importlib


def patch_with_timer(path):
    agent_module = importlib.import_module(path)
    def new_act():

        start = time.time()
        agent_module.act()
        end = time.time()
        return end - start

    mock = Mock()
    mock(path, new_act)



