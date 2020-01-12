# Environment setup

1. Run `pip install -r requirements.txt`

# Development tools

- Run an instance of a game: `tools/run.ipynb`.

    - Debug individual steps: `tools/debug.py`.

- Evaluate agent performance: `tools/evaluate.py`.

# Deployment

To upload `agents/selects_leftmost.py` agent:

1. Commit all changes made to the repository to the `master` branch
2. Run `$ ./deploy.sh selects_leftmost`

Note, there's a limit of 2 submissions per 24 hours.