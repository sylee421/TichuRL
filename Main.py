from tichu.Env import Env
from agents.Random import Random
from agents.Human import Human
from agents.Priority_min import Priority_min
from agents.DQN import Agent
import tensorflow as tf

config = tf.ConfigProto()

### Set environmets
env = Env(human=0, verbose=0)
episode_num = 100

with tf.Session(config=config) as sess:
    ### Set agents
    agent_0 = Priority_min()
    agent_1 = Priority_min()
    agent_2 = Priority_min()
    agent_3 = Agent(sess, is_training=True)
    env.set_agents([agent_0, agent_1, agent_2, agent_3])

    ### Run
    for episode in range(episode_num):
        env.run(is_training=True)
    agent_3.save()
    env.verbose = 1
    env.human = 1
    agent_3.load()
    for episode in range(episode_num):
        env.run(is_training=False)

