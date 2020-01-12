from kaggle_environments import make

from agents import selects_leftmost

AGENT = selects_leftmost.act

if __name__ == '__main__':
    env = make('connectx', debug=True)
    trainer = env.train([None, 'negamax'])
    observation = trainer.reset()
    while not env.done:
        my_action = AGENT(observation, env.configuration)
        print("My Action", my_action)
        observation, reward, done, info = trainer.step(my_action)
