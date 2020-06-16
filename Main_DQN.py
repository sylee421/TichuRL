import tensorflow as tf
import numpy as np

from tichu.Env import Env
from agents.Random import Random
from agents.Human import Human
from agents.Priority_min import Priority_min
from agents.DQN_SY import DQNAgent
from agents.handValueNet import HandValueNet

##########DEBUG
#from tichu.Card import Cards
#from tichu.Card import Card
#test = Cards(card_list=[Card('3','Spade'), Card('4','Spade'), Card('5','Spade'), Card('6','Spade'), Card('7','Heart')])
#test.set_combination()
#print(test.type)
#print(test.value)
##############

### Set environmets
env = Env(human=0, verbose=0)
eval_env = Env(human=0, verbose=1)

### Set parameters
evaluate_every = 100
episode_num = 1000
memory_init_size = 1000
train_every = 1
learning_rate = 0.00001

### Config
#config = tf.compat.v1.ConfigProto(device_count = {'GPU':0}) # gpu off

#with tf.compat.v1.Session(config=config) as sess:
with tf.compat.v1.Session() as sess:

    ### Set up agents
    agent_0 = DQNAgent(sess,
                       scope='dqn',
                       action_num=env.action_num,
                       replay_memory_size=20000,
                       replay_memory_init_size=memory_init_size,
                       train_every=train_every,
                       state_shape=env.state_shape,
                       mlp_layers=[512,512],
                       learning_rate=learning_rate)
    agent_1 = Random()
    agent_2 = Random()
    agent_3 = Random()
    env.set_agents([agent_0, agent_1, agent_2, agent_3])
    eval_env.set_agents([agent_0, agent_1, agent_2, agent_3])

    ### Initialize
    sess.run(tf.compat.v1.global_variables_initializer())
    points = np.zeros((4,), dtype=int)

    agent_0.load()

    ### Run
    for episode in range(episode_num):
        trajectories, point = env.run(is_training=True)
        points = np.add(points,point)

        for player in range(4):
            for ts in trajectories[player]:
                agent_0.feed(ts)

        if episode% evaluate_every == 0:
            eval_env.run(is_training=False)
            print(points)
            points = np.zeros((4,), dtype=int)

    agent_0.save()
