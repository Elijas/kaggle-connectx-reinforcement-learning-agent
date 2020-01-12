from kaggle_environments import make

from bots import always_select_first

AGENT = always_select_first.act

if __name__ == '__main__':
    env = make('connectx', debug=True)
    trainer = env.train([None, 'negamax'])
    observation = trainer.reset()
    while not env.done:
        my_action = AGENT(observation, env.configuration)
        print("My Action", my_action)
        observation, reward, done, info = trainer.step(my_action)
