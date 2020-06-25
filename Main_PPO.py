import tensorflow as tf
import numpy as np

from tichu.Env import Env
from agents.Random import Random
from agents.Human import Human
from agents.Priority_min import Priority_min
from agents.PPO_SY import PPOAgent
from agents.handValueNet import HandValueNet

### Set environmets
env = Env(human=0, verbose=1)
eval_env = Env(human=0, verbose=1)

### Set parameters
evaluate_every = 1
episode_num = 10
train_every = 1
learning_rate = 0.00025

### Config
#config = tf.compat.v1.ConfigProto(device_count = {'GPU':0}) # gpu off

#with tf.compat.v1.Session(config=config) as sess:
with tf.compat.v1.Session() as sess:

    ### Set up agents
    agent_0 = PPOAgent(sess)
    agent_1 = Random()
    agent_2 = Random()
    agent_3 = Random()
    env.set_agents([agent_0, agent_1, agent_2, agent_3])
    eval_env.set_agents([agent_0, agent_1, agent_2, agent_3])

    ### Initialize
    sess.run(tf.compat.v1.global_variables_initializer())
    points = np.zeros((4,), dtype=int)

#    agent_0.load()

    ### Run
    for episode in range(episode_num):
        trajectories, point = env.run(is_training=True)
        points = np.add(points,point)

        agent_0.feed(trajectories[0])

        if episode% evaluate_every == 0:
            print(points)
            points = np.zeros((4,), dtype=int)

#    agent_0.save()
