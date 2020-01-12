from kaggle_environments import make

from agents import selects_leftmost

AGENT_1 = selects_leftmost.act
AGENT_2 = 'negamax'

if __name__ == '__main__':
    env = make('connectx', debug=True)
    trainer = env.train([None, AGENT_2])
    observation = trainer.reset()
    while not env.done:
        my_action = AGENT_1(observation, env.configuration)
        print("My Action", my_action)
        observation, reward, done, info = trainer.step(my_action)
