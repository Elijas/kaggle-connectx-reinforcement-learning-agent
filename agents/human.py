def act(observation, configuration):
    return int(input(f"Enter action number [1-{configuration['columns']}]: ")) - 1
