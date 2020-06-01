import tensorflow as tf
import of

from tichu.Env import Env
from agents.Random import Random
from agents.Human import Human
from agents.Priority_min import Priority_min
from agents.DQN_SY import DQNAgent

### Set environmets
env = Env(human=0, verbose=0)
eval_env = Env(human=0, verbose=1)

### Set parameters
evaluate_every = 100
episode_num = 1000
memory_init_size = 100
train_every = 1

with tf.Session() as sess:

    ### Set up agents
    agent_0 = DQNAgent(sess,
                       scope='dqn',
#                       action_num=env.action_num,
                       replay_memory_size=20000,
                       replay_memory_init_size=memory_init_size,
                       train_every=train_every,
#                       state_shape=env.state_shape,
                       mlp_layers=[512,512])
    agent_1 = Random()
    agent_2 = Random()
    agent_3 = Random()
    env.set_agents([agent_0, agent_1, agent_2, agent_3])
    eval_env.set_agents([agent_0, agent_1, agent_2, agent_3])

    ### Initialize
    sess.run(tf.global_variables_initializer())

    ### Run
    for episode in range(episode_num):
        trajectories, _ = env.run(is_training=True)

        for ts in trajectories[0]:
            agent_0.feed(ts)

        if episode% evaluate_every == 0:
            eval_env.run(is_training=False)


save_dir = './dqn/'
if not os.path.exits(save_dir):
    os.makedirs(svae_dir)
saver = tf.train.Saver()
saver.save(sess, os.path.join(save_dir, 'model'))

