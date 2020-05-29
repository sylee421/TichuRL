import sys
import os
from collections import deque
import random
import time
import gym
import numpy as np
import argparse
import tensorflow as tf

class ReplayMemory(object):
    # data를 저장할 memory를 만듬, batch size 저장
    def __init__(self, state_size, batch_size):
        self.memory = deque(maxlen=2000)
        self.state_size = state_size
        self.batch_size = batch_size
        pass

    # (state, action, reward, next state, terminal의 data를 저장)
    def add(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))
        pass

    # replay memory에서 batch size 만큼 개수의 data를 random하게 return
    def mini_batch(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, terminals = [], [], []
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            next_states[i] = mini_batch[i][3]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            terminals.append(mini_batch[i][4])
        return states, actions, rewards, next_states, terminals
        pass

class DQN(object):
    # sess, replay memory 저장, mini batch로 입력 받을 state와 action과 Q target을 위한 placeholder 생성
    # 학습시킬 Q nerwork와 학습 때 필요한 target network를 생성하고 두 network의 ourput을 각각 저장
    # 학습을 위한 oprimizaer 생성
    def __init__(self, state_size, action_size, sess, learning_rate, replay, discount_factor):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess
        self.lr = learning_rate
        self.replay = replay
        self.discount_factor = discount_factor

        self.states = tf.placeholder(tf.float32, [None, self.state_size])
        self.actions = tf.placeholder(tf.int64, [None])
        self.target = tf.placeholder(tf.float32, [None])

        self.prediction_Q = self.build_network('pred')
        self.target_Q = self.build_network('target')
        self.train_op = self.build_optimizer()
        pass
    
    # 학습시킬 Q nerwork('pred')와 학습에 필요한 target network('target')을 Fully connected network로 만듬
    def build_network(self, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            h1 = tf.layers.dense(self.states, 25, activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal)
            h2 = tf.layers.dense(h1, 125, activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal)
            h3 = tf.layers.dense(h2, 25, activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal)
            output = tf.layers.dense(h3, self.action_size, kernel_initializer=tf.initializers.truncated_normal)

            return output
            pass
    # action을 one hot으로 바꾼 후 Q network output과 곱하면 원하는 Q(s,a)만 남는다(q_value)
    # loss는 Train Network에서 미리 계산된 Q target값과 q_value값의 mse loss로 계산됨 loss값과 이를 최소화하는 optimizer를 return
    def build_optimizer(self):
        actions_one_hot = tf.one_hot(self.actions, self.action_size, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(actions_one_hot, self.prediction_Q), axis=1)
        loss = tf.reduce_mean(tf.square(self.target - q_value))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return loss, train_op
        pass
    # replay memory에서 mini batch를 받는다, Q target 값을 계산하여 Build Optimizer 실행, 실질적으로 train을 담당하는 function
    def train_network(self):
        states, actions, rewards, next_states, terminals = self.replay.mini_batch()

        target_Q = self.sess.run(self.target_Q, feed_dict={self.states: next_states})
        target = []
        for i in range(self.replay.batch_size):
            if terminals[i]:
                target.append(rewards[i])
            else:
                target.append(rewards[i] + self.discount_factor * np.max(target_Q[i]))
        
        self.sess.run(self.train_op, feed_dict={self.states: states, self.actions: actions, self.target: target})
        pass
    # 우리가 학습시키는 Q network를 target network로 복사하는 function
    def update_target_network(self):
        copy_op = []
        pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
        for pred_var, target_var in zip(pred_vars, target_vars):
            copy_op.append(target_var.assign(pred_var.value()))
        self.sess.run(copy_op)
    # Q network의 output을 return
    def predict_Q(self, states):
        return self.sess.run(self.prediction_Q, feed_dict={self.states:states})
        pass

class Agent(object):
    def __init__(self, sess, is_training=True):
        # cartpole 환경
        self.is_training = is_training
        if self.is_training:
            self.eps = 1.0 # 초기 epsilon 값 생성
        else: 
            self.eps = 0
        self.sess = sess
        self.state_size = 12
        self.action_size = 15
        self.epsilon_decay_steps = 1e2 # epsilone이 0.1이 되기까지의 time step 수
        self.learning_rate = 0.001
        self.batch_size = 4
        self.discount_factor = 0.99
        self.replay = ReplayMemory(self.state_size, self.batch_size)
        self.dqn = DQN(self.state_size, self.action_size, self.sess, self.learning_rate, self.replay, self.discount_factor)
        self.saver = tf.train.Saver()
        # to do add argument
        self.sess.run(tf.global_variables_initializer())
        self.dqn.update_target_network()
        self.score = 0
        self.scores = []
        self.episode = 0
        pass

    # epsilon greedy에 따라 최적의 action을 return
    def select_action(self, state):
        if np.random.rand() <= self.eps:
            return random.choice(state['legal_actions'])
        else:
            state_in = self.state_input(state)
            q_value = self.dqn.predict_Q(state_in)
            # print("q_value : ", q_value[0])

            # find the argmax of q_value with constraint of legal actions
            actions = []
            for item in state['legal_actions']:
                if item.value < 16:
                    # print("q_value, action : ", item.value, q_value[0][item.value])
                    actions.append((item.value, q_value[0][item.value]))
                print("action, q_value", actions)
            max_idx = np.argmax(actions, axis=0)[1]
            for item in state['legal_actions']:
                if actions[max_idx][0] == item.value:
                    return item
        pass

    def step(self, state):
        action = self.select_action(state)
        print("final action : ", action.value)
        return action

    def state_input(self, state):
        # parsing the ground type into the int number
        if state['ground'].type == 'none':
            gr_type = 0
        elif state['ground'].type == 'solo':
            gr_type = 1
        elif state['ground'].type == 'pair':
            gr_type = 2
        elif state['ground'].type == 'triple':
            gr_type = 3
        elif state['ground'].type == 'four':
            gr_type = 4
        elif state['ground'].type == 'full':
            gr_type = 5
        elif state['ground'].type == 'strat':
            gr_type = 6
        else:
            raise ValueError("[get_legal_combination] Wrong ground type")
        
        # parsing the hand available combination type
        type_solo = 0
        type_pair = 0
        type_triple = 0
        type_four = 0
        type_full = 0
        type_strat = 0
        if state['hand'].get_available_combination()[0]:
            type_solo = 1
        if state['hand'].get_available_combination()[1]:
            type_pair = 1
        if state['hand'].get_available_combination()[2]:
            type_triple = 1
        if state['hand'].get_available_combination()[3]:
            type_four = 1
        if state['hand'].get_available_combination()[4]:
            type_full = 1
        if state['hand'].get_available_combination()[5]:
            type_strat = 1

        # print(gr_type, type_solo, type_pair, type_triple, type_four, type_full, type_strat)        
        # print(state['hand'].size, state['ground'].value, state['card_num'][0], state['card_num'][1], state['card_num'][2])
        # state['hand'].show()

        # reshape the state fitted in DQN
        state = np.array([state['hand'].size, state['ground'].value, gr_type, type_solo, state['card_num'][0], state['card_num'][1], state['card_num'][2], type_pair, type_triple, type_four, type_full, type_strat])
        # state = np.array([state['hand'].size, state['ground'].value, gr_type])
        state = np.reshape(state, [1, self.state_size])
        return state

    def DQN_input(self, state, next_state, reward, terminal, action):
        state = self.state_input(state)
        next_state = self.state_input(next_state)
        act_val = action.value
        return state, next_state, act_val

    def train_d(self, state, next_state, reward, terminal, action):
        print("reward", reward)
        state_in, next_state_in, action_in = self.DQN_input(state, next_state, reward, terminal, action)
        self.replay.add(state_in, action_in, reward, next_state_in, terminal)
        if len(self.replay.memory) >= 10:
            if self.eps > 0.01:
                self.eps -= 0.9 / self.epsilon_decay_steps
            self.dqn.train_network()        
        self.score += reward

        if terminal:
            print("terminal raise, update target network")
            self.dqn.update_target_network()
            self.episode = self.episode + 1
            print('episode: ', self.episode, 'score: ', int(self.score), 'epsilon ', self.eps)
            self.score = 0

    # 학습한 weights를 save하고 load함
    def save(self):
        checkpoint_dir = 'dqn_tensorflow'
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))
    
    def load(self):
        checkpoint_dir = 'dqn_tensorflow'
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))
        

if __name__ == "__main__":

    # parameter 저장하는 parser
    parser = argparse.ArgumentParser(description="CartPole")
    parser.add_argument('--env_name', default='CartPole-v1', type=str)
    parser.add_argument('--epsilon_decay_steps', default=1e4, type=int, help="how many steps for epsilon to be 0.1")
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--episodes', default=500, type=float)
    sys.argv = ['-f']
    args = parser.parse_args()

    config = tf.ConfigProto()
    #os.environ["CUDA_VISIBLA_DEVICES"] = '0' # 선택한 gpu에만 메모리 할당
    #config.log_device_placement = False # 디바이스 배치결과 보여주느지 여부
    #config.gpu_options.allow_growth = True # 필요에 따라 탄력적으로 메모리 사용

    with tf.Session(config=config) as sess:
        agent = Agent(args, sess)
        sess.run(tf.global_variables_initializer())
        agent.train()
        agent.save()
        agent.load()
        rewards = []
        for i in range(20):
            r = agent.play()
            rewards.append(int(r))
        mean = np.mean(rewards)
        print(rewards)
        print(mean)

## Appendix Old codes

# class Environment(object):
#     # env(gym env), state size와 action size를 저장
#     def __init__(self, env, state_size, action_size):
#         self.env = env
#         self.state_size = state_size
#         self.action_size = action_size
#         pass
#     # env의 random action을 return
#     def random_action(self):
#         return random.randrange(self.action_size)
#         pass
#     # 실제 작동하는 gym의 env 모습을 보여줌
#     def render_worker(self, render):
#         if render :
#             self.env.render()
#         pass
#     # 새로운 episode를 시작
#     def new_episode(self):
#         state = self.env.reset()
#         state = np.reshape(state, [1, self.state_size])
#         return state
#         pass
#     # action을 받고 그것을 실행하여 next state와 reward를 return
#     def act(self, action):
#         next_state, reward, terminal, _ = self.env.step(action)
#         return next_state, reward, terminal
#         pass

# def train(self):
#     scores, episodes = [], []
#     self.dqn.update_target_network()

#     for e in range(self.episodes):
#         terminal = False
#         score = 0
#         state = self.ENV.new_episode()

#         while not terminal: # episode가 끝났을 때
#             action = self.select_action(state)
#             next_state, reward, terminal = self.ENV.act(action)
#             next_state = np.reshape(next_state, [1, self.state_size])
#             self.replay.add(state, action, reward, next_state, terminal)

#             if len(self.replay.memory) >= 1000:
#                 if self.eps > 0.01:
#                     self.eps -= 0.9 / self.epsilon_decay_steps
#                 self.dqn.train_network()
            
#             score += reward
#             state = next_state

#             if terminal:
#                 self.dqn.update_target_network()
#                 scores.append(score)
#                 episodes.append(e)
#                 print('episode: ', e, 'score: ', int(score), 'epsilon ', self.eps, 'last 10 mean score ', np.mean(scores[-min(10, len(scores)):]))

#                 if np.mean(scores[-min(10, len(scores)):]) > self.env._max_episode_steps * 0.95:
#                     print('Already well trained')
#                     return
#     pass
# test 하는 function, epsilon 0로 random을 없애고 실행
# render worker를 통해 어떻게 작동 되는지 직접 볼 수 있음
# def play(self):
#     state = self.ENV.new_episode()
#     self.eps = 0.0

#     terminal = False
#     score = 0
#     while not terminal:
#         self.ENV.render_worker(True)
#         action = self.select_action(state)
#         next_state, reward, terminal = self.ENV.act(action)
#         next_state = np.reshape(next_state, [1, self.state_size])
#         score += reward
#         state = next_state
#         time.sleep(0.02)
#         if terminal:
#             time.sleep(1)
#             return score
#     pass




