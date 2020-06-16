import tensorflow as tf
import os

from tichu.Env import Env
from agents.Random import Random
from agents.Human import Human
from agents.Priority_min import Priority_min
from agents.DQN_SY import DQNAgent
from agents.handValueNet import HandValueNet

### Set environmets
env = Env(human=1, verbose=1)

### Set parameters
episode_num = 1

### Config
config = tf.ConfigProto(device_count = {'GPU':0})

with tf.compat.v1.Session(config=config) as sess:

    ### Set up agents
    agent_0 = Human()
    agent_1 = Random()
    agent_2 = Random()
    agent_3 = Random()
    env.set_agents([agent_0, agent_1, agent_2, agent_3])

    ### Initialize
    sess.run(tf.global_variables_initializer())

    ### Run
    for episode in range(episode_num):
        env.run(is_training=False)

