# Environment setup

1. Run `pip install -r requirements.txt`

# Development tools

- Render an instance of a game: `tools/run.ipynb`

    - Debug individual steps with a function: `tools/debug.py`
    - Play against an agent yourself: `tools/debug_manual.py`
    
- Evaluate agent performance: `tools/evaluate.py`

# Deployment

To upload `agents/selects_leftmost.py` agent:

1. Commit all changes made to the repository to the `master` branch
2. Run `$ ./deploy.sh selects_leftmost`

Note, there's a limit of 2 submissions per 24 hours.

# Creating a new agent

1. Create a file in `./agents` folder.
2. Add the agent's act function to the `AGENTS` array (`./tools/evaluate.py`).

# Additional notes

The board size and winning condition may change in the future. Add an `assert` at the top of an agent function if the agent is depending on the conditions. For example, to play only in games with the default configuration, use:
```
assert configuration.columns == 7 \
and configuration.rows == 6 \
and configuration.inarow == 4 \
and configuration.timeout >= 5
```
