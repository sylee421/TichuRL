import sys
import os
from collections import deque
import random
import time
import gym
import numpy as np
import argparse
import tensorflow as tf

class Environment(object):
    # env(gym env), state size와 action size를 저장
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        pass
    # env의 random action을 return
    def random_action(self):
        return random.randrange(self.action_size)
        pass
    # 실제 작동하는 gym의 env 모습을 보여줌
    def render_worker(self, render):
        if render :
            self.env.render()
        pass
    # 새로운 episode를 시작
    def new_episode(self):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        return state
        pass
    # action을 받고 그것을 실행하여 next state와 reward를 return
    def act(self, action):
        next_state, reward, terminal, _ = self.env.step(action)
        return next_state, reward, terminal
        pass

class ReplayMemory(object):
    # data를 저장할 memory를 만듬, batch size 저장
    def __init__(self, env, state_size, batch_size):
        self.memory = deque(maxlen=2000)
        self.env = env
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
            h2 = tf.layers.dense(h1, 25, activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal)
            output = tf.layers.dense(h2, self.action_size, kernel_initializer=tf.initializers.truncated_normal)

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
    def __init__(self, args, sess):
        # cartpole 환경
        self.env = gym.make(args.env_name)
        self.eps = 1.0 # 초기 epsilon 값 생성
        self.sess = sess
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.env._max_episode_steps = 500
        self.epsilon_decay_steps = args.epsilon_decay_steps # epsilone이 0.1이 되기까지의 time step 수
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.discount_factor = args.discount_factor
        self.episodes = args.episodes
        self.ENV = Environment(self.env, self.state_size, self.action_size)
        self.replay = ReplayMemory(self.env, self.state_size, self.batch_size)
        self.dqn = DQN(self.state_size, self.action_size, self.sess, self.learning_rate, self.replay, self.discount_factor)
        self.saver = tf.train.Saver()
        pass
    # epsilon greedy에 따라 최적의 action을 return
    def select_action(self, state):
        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)
        else:
            q_value = self.dqn.predict_Q(state)
            return np.argmax(q_value[0])
        pass
    # train이 시작되고 replay memory에 일정 수 만큼 data가 쌓인 후 train을 시작
    # score는 episode의 reward의 합으로 return을 뜻함
    # 꼭 1개의 episode가 끝날 때마다 target network update를 할 필요는 없음
    # 최근 10개의 episode의 score이 원하는 값을 넘으면 train을 빨리 끝냄
    def train(self):
        scores, episodes = [], []
        self.dqn.update_target_network()

        for e in range(self.episodes):
            terminal = False
            score = 0
            state = self.ENV.new_episode()

            while not terminal: # episode가 끝났을 때
                action = self.select_action(state)
                next_state, reward, terminal = self.ENV.act(action)
                
                next_state = np.reshape(next_state, [1, self.state_size])
                print("next_state : ", next_state)
                self.replay.add(state, action, reward, next_state, terminal)

                if len(self.replay.memory) >= 1000:
                    if self.eps > 0.01:
                        self.eps -= 0.9 / self.epsilon_decay_steps
                    self.dqn.train_network()
                score += reward
                state = next_state

                if terminal:
                    self.dqn.update_target_network()
                    scores.append(score)
                    episodes.append(e)
                    print('episode: ', e, 'score: ', int(score), 'epsilon ', self.eps, 'last 10 mean score ', np.mean(scores[-min(10, len(scores)):]))

                    if np.mean(scores[-min(10, len(scores)):]) > self.env._max_episode_steps * 0.95:
                        print('Already well trained')
                        return
        pass
    # test 하는 function, epsilon 0로 random을 없애고 실행
    # render worker를 통해 어떻게 작동 되는지 직접 볼 수 있음
    def play(self):
        state = self.ENV.new_episode()
        self.eps = 0.0

        terminal = False
        score = 0
        while not terminal:
            self.ENV.render_worker(True)
            action = self.select_action(state)
            next_state, reward, terminal = self.ENV.act(action)
            next_state = np.reshape(next_state, [1, self.state_size])
            score += reward
            state = next_state
            time.sleep(0.02)
            if terminal:
                time.sleep(1)
                return score
        pass
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





