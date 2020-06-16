import tensorflow as tf

from tichu.Card import Cards
from tichu.Card import Card
from tichu.Env import Env
from agents.Random import Random
from agents.Human import Human
from agents.Priority_min import Priority_min
from agents.DQN_SY import DQNAgent
from agents.handValueNet import HandValueNet

### Set environmets
env = Env(human=0, verbose=0)
eval_env = Env(human=0, verbose=0)

### Set parameters
evaluate_every = 1000
episode_num = 1000
memory_init_size = 100
train_every = 1

with tf.compat.v1.Session() as sess:

    ### Set up agents
    agent_0 = HandValueNet(sess)
    agent_1 = Random()
    agent_2 = Random()
    agent_3 = Random()
    env.set_agents([agent_0, agent_1, agent_2, agent_3])
    eval_env.set_agents([agent_0, agent_1, agent_2, agent_3])

    ### Initialize
    sess.run(tf.global_variables_initializer())
    agent_0.load()

    ### Run
    for episode in range(episode_num):
        hand, point = env.run(is_training=True)

        agent_0.feed(hand, point)

        if episode% evaluate_every == 0:
            eval_hand, eval_point = eval_env.run(is_training=False)
            pred = agent_0.predict(eval_hand)
            print("Hand : ")
            eval_hand.show()
            print("pred_point : %f" %(pred))
            print("point : %d" %(eval_point[0]))
            

#    hand = Cards([Card('5','Spade'),
#            Card('Q','Heart'), Card('Q','Spade'), Card('Q','Dia'),Card('Q','Club'),
#            Card('K','Heart'), Card('K','Spade'), Card('K','Dia'),Card('K','Club'),
#            Card('A','Heart'), Card('A','Spade'), Card('A','Dia'),Card('A','Club')])
#    hand.show()
#    pred = agent_0.predict(hand)
#    print("pred_point : %f" %(pred))

    agent_0.save()
