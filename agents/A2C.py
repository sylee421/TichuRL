import sys
import os
from collections import deque
import random
import time
import gym
import numpy as np
import argparse
import tensorflow as tf

# A2C adaptng.
# states : hand size, ground value, ground type, others1 card size, others2 card size, others3 card size, cards values 
# actions : hand cards index

class ReplayMemory(object):
    # data를 저장할 memory를 만듬, batch size 저장
    def __init__(self, state_size, batch_size):
        self.memory = deque(maxlen=2000)
        self.state_size = state_size
        self.batch_size = batch_size
        pass

    # (state, action, reward, next state, terminal의 data를 저장)
    def add(self, state, action, reward, next_state, terminal, leg_act_onehot):
        self.memory.append((state, action, reward, next_state, terminal, leg_act_onehot))
        pass

    # replay memory에서 batch size 만큼 개수의 data를 random하게 return
    def mini_batch(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, terminals, leg_act_onehots = [], [], [], []
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            next_states[i] = mini_batch[i][3]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            terminals.append(mini_batch[i][4])
            leg_act_onehots.append(mini_batch[i][5])
        return states, actions, rewards, next_states, terminals, leg_act_onehots
        pass

class A2C(object):
    # sess, replay memory 저장, mini batch로 입력 받을 state와 action과 Q target을 위한 placeholder 생성
    # 학습시킬 Q nerwork와 학습 때 필요한 target network를 생성하고 두 network의 ourput을 각각 저장
    # 학습을 위한 oprimizaer 생성
    def __init__(self, state_size, action_size, value_size, sess, learning_rate, replay, discount_factor):
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = value_size
        self.sess = sess
        self.lr = learning_rate
        self.replay = replay
        self.discount_factor = discount_factor
        self.value_factor = 0.3
        self.entropy_factor = 0.001

        self.states = tf.placeholder(tf.float32, [None, self.state_size])
        self.actions = tf.placeholder(tf.int64, [None])
        self.target = tf.placeholder(tf.float32, [None, self.value_size])
        self.leg_act_onehots = tf.placeholder(tf.float32, [None, self.action_size])

        self.prediction_P, self.prediction_V = self.build_network('pred')
        self.target_P, self.target_V = self.build_network('target')
        self.train_op = self.build_optimizer()
        self.loss = 0

        pass
    
    # 학습시킬 Q nerwork('pred')와 학습에 필요한 target network('target')을 Fully connected network로 만듬
    def build_network(self, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            h1 = tf.layers.dense(self.states, 256, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, 512, activation=tf.nn.relu)
            h3 = tf.layers.dense(h2, 512, activation=tf.nn.relu)
            policy = tf.layers.dense(h3, self.action_size)
            value = tf.layers.dense(h3, self.value_size)

            return policy, value
            pass

    # action을 one hot으로 바꾼 후 Q network output과 곱하면 원하는 Q(s,a)만 남는다(q_value)
    # loss는 Train Network에서 미리 계산된 Q target값과 q_value값의 mse loss로 계산됨 loss값과 이를 최소화하는 optimizer를 return
    def build_optimizer(self):
        actions_one_hot = tf.one_hot(self.actions, self.action_size, 1.0, 0.0)

        leg_prediction_P = tf.nn.softmax(self.prediction_P)
        leg_prediction_P = leg_prediction_P * self.leg_act_onehots
        leg_prediction_P = tf.nn.softmax(leg_prediction_P)

        log_policy = tf.log(tf.clip_by_value(leg_prediction_P, 1e-20, 1.0))
        log_prob = tf.reduce_sum(log_policy * actions_one_hot, axis=1)

        value_loss = tf.reduce_mean(tf.square(self.target  - self.prediction_V))
        value_loss *= self.value_factor

        entropy = -tf.reduce_mean(tf.reduce_sum(leg_prediction_P * log_policy, axis=1))
        entropy *= self.entropy_factor
        policy_loss = tf.reduce_mean(log_prob * tf.subtract(self.target, self.prediction_V))

        loss = value_loss - policy_loss - entropy
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        return loss, train_op
        pass

    # replay memory에서 mini batch를 받는다, Q target 값을 계산하여 Build Optimizer 실행, 실질적으로 train을 담당하는 function
    def train_network(self):
        states, actions, rewards, next_states, terminals, leg_act_onehots = self.replay.mini_batch()

        target_V = self.sess.run(self.prediction_V, feed_dict={self.states: next_states})

        # calculate next state q_value based on Q function        
        target = []
        for i in range(self.replay.batch_size):
            if terminals[i]:
                target.append([rewards[i]])
            else:
                target.append([rewards[i] + self.discount_factor * target_V[i][0]])

        self.loss, _ = self.sess.run(self.train_op, feed_dict={self.states: states, self.actions: actions, self.target: target, self.leg_act_onehots: leg_act_onehots})
        pass

    # 우리가 학습시키는 Q network를 target network로 복사하는 function
    def update_target_network(self):
        copy_op = []
        pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
        for pred_var, target_var in zip(pred_vars, target_vars):
            copy_op.append(target_var.assign(pred_var.value()))
        self.sess.run(copy_op)

    # Policy network의 output을 return
    def predict_P(self, states):
        return self.sess.run(self.prediction_P, feed_dict={self.states:states})
        pass

class Agent_A2C(object):
    def __init__(self, sess, is_training=True):
        # cartpole 환경
        self.is_training = is_training
        if self.is_training:
            self.eps = 1.0 # 초기 epsilon 값 생성
        else: 
            self.eps = 0
        self.sess = sess
        self.state_size = 19
        self.action_size = 2**13
        self.value_size = 1
        self.epsilon_decay_steps = 200000 # epsilon이 0.1이 되기까지의 time step 수
        self.learning_rate = 0.00005
        self.batch_size = 64
        self.discount_factor = 0.9
        self.replay = ReplayMemory(self.state_size, self.batch_size)
        self.a2c = A2C(self.state_size, self.action_size, self.value_size, self.sess, self.learning_rate, self.replay, self.discount_factor)
        self.saver = tf.train.Saver()
        # to do add argument
        self.sess.run(tf.global_variables_initializer())
        self.a2c.update_target_network()
        self.score = 0
        self.episode = 0
        self.mean_loss = 0
        pass

    def action2Qidx(self, card_set, item):
        q_idx = 0
        if item.cards:
            for item2 in item.cards:
                idx = card_set.cards.index(item2)
                q_idx += 2 ** idx
        return q_idx

    def idx2onehot(self, idx, size):
        targets = idx
        one_hot_target = np.eye(size)[targets]
        return one_hot_target

    def legal_act_onehot(self, state):
        # parsing the legal actions to Q value index
        q_idx_onehot = np.zeros(self.action_size)
        for item in state['legal_actions']:
            q_idx = self.action2Qidx(state['hand'], item)
            q_idx_onehot += self.idx2onehot(q_idx, self.action_size) # need to make function
        return q_idx_onehot

    # epsilon greedy에 따라 최적의 action을 return
    def select_action(self, state):
        # epsilon greedy action select
        if (np.random.rand() <= self.eps):
            return random.choice(state['legal_actions'])
        else:
            state_in = self.state_input(state)
            q_value = self.a2c.predict_P(state_in)
            
            q_idx_onehot = self.legal_act_onehot(state)

            # filter out the legal actions
            legal_q_value = q_value[0] * q_idx_onehot
            max_q = np.max(legal_q_value[legal_q_value!=0])
            max_q_idx = np.where(legal_q_value==max_q)[0][0]

            # return the maximum item based on the hand cards
            for item in state['legal_actions']:
                q_idx = self.action2Qidx(state['hand'], item)
                if q_idx == max_q_idx:
                    return item
        pass

    def step(self, state):
        action = self.select_action(state)
        return action

    def state_input(self, state):
        # parsing the ground type into the int number
        if state['ground'].type == 'none':
            gr_type = 6
        elif state['ground'].type == 'solo':
            gr_type = 0
        elif state['ground'].type == 'pair':
            gr_type = 1
        elif state['ground'].type == 'triple':
            gr_type = 2
        elif state['ground'].type == 'four':
            gr_type = 3
        elif state['ground'].type == 'full':
            gr_type = 4
        elif state['ground'].type == 'strat':
            gr_type = 5
        elif state['ground'].type == 'strat_flush':
            gr_type = 7
        elif state['ground'].type == 'pair_seq':
            gr_type = 8
        else:
            raise ValueError("[get_legal_combination] Wrong ground type")

        # temporary filter the strat value that is over 14
        if state['ground'].value > 100:
            gr_val = state['ground'].value % 1000
        else:
            gr_val = state['ground'].value

        # make state input with reference information (hand size, ground type value, other player card number)
        state_rt = np.array([state['hand'].size, gr_val, gr_type, state['card_num'][0], state['card_num'][1], state['card_num'][2]])

        # append the state with cards value
        for i in range(state['hand'].num_show):
            if i < len(state['hand'].cards):
                state_rt = np.append(state_rt, state['hand'].cards[i].value)
            else:
                state_rt = np.append(state_rt, 0)

        # reshape the state fitted in A2C
        state_rt = np.reshape(state_rt, [1, self.state_size])
        return state_rt

    def A2C_input(self, state, next_state, reward, terminal, action):
        state_rt = self.state_input(state)
        next_state_rt = self.state_input(next_state)
        leg_act_onehot = self.legal_act_onehot(state)

        # change action value if the action is pass        
        if action.size == 0:
            act_val = 0
        else:
            act_val = self.action2Qidx(state['hand'], action)

        return state_rt, next_state_rt, act_val, leg_act_onehot

    def train_d(self, state, next_state, reward, terminal, action):
        state_in, next_state_in, action_in, leg_act_onehot = self.A2C_input(state, next_state, reward, terminal, action)
        self.replay.add(state_in, action_in, reward, next_state_in, terminal, leg_act_onehot)
        if len(self.replay.memory) >= 1000:
            if self.eps > 0.01:
                self.eps -= 0.9 / self.epsilon_decay_steps
            self.a2c.train_network()
        self.score += reward

        if terminal:
            self.episode = self.episode + 1
            if self.episode % 100 == 0:
                print('episode: ', self.episode, 'score: ', int(self.score), 'loss: ' , self.mean_loss/10, 'cur loss : ', self.a2c.loss, 'epsilon ', self.eps)
                self.mean_loss = 0
            elif self.episode % 100 >= 90 :
                self.mean_loss += self.a2c.loss
            self.score = 0

    # 학습한 weights를 save하고 load함
    def save(self):
        checkpoint_dir = 'a2c_tensorflow'
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))
    
    def load(self):
        checkpoint_dir = 'a2c_tensorflow'
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


