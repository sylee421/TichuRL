import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 선택한 gpu에만 메모리 할당

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

import numpy as np
from tichu.Env_old import Env
from agents.Random import Random
from agents.Human import Human
from agents.Priority_min import Priority_min
from agents.double_DQN5 import Agent_2DQN
# from agents.A2C import Agent_A2C

config = tf.ConfigProto()

### Set environmets
env = Env(human=0, verbose=0)
episode_num = 5000
val_episode = 1000

config.log_device_placement = False # 디바이스 배치결과 보여주느지 여부
config.gpu_options.allow_growth = True # 필요에 따라 탄력적으로 메모리 사용

with tf.Session(config=config) as sess:
    ### Set agents training
    agent_0 = Priority_min()
    agent_1 = Priority_min()
    agent_2 = Priority_min()
    agent_3 = Agent_2DQN(sess, is_training=True)
    env.set_agents([agent_0, agent_1, agent_2, agent_3])
    # agent_3.load()
    ### Run
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for episode in range(episode_num):
        agent_3.is_training = True
        env.run(is_training=True)
        if episode % 1000 == 0:
            agent_3.save()
            agent_3.is_training = False
            env.points = np.zeros(4)
            env.verbose = 0
            env.human = 0            
            for val_game in range(val_episode):
                env.run(is_training=False)
            print("episode : ", episode," Point: " + str(env.points))

    coord.request_stop()
    coord.join(threads)

    # for Evaluation
    # # Set agents evaluation with loading
    # agent_0 = Priority_min()
    # agent_1 = Priority_min()
    # agent_2 = Priority_min()
    # agent_3 = Agent_2DQN(sess, is_training=False)
    # env.set_agents([agent_0, agent_1, agent_2, agent_3])
    # env.verbose = 0
    # env.human = 0        
    # agent_3.load()
    # for episode in range(val_episode):
    #     env.run(is_training=False)
    #     if episode % 100 == 0:
    #         print("episode : ", episode," Point: " + str(env.points))


