from collections import namedtuple
import numpy as np
import tensorflow as tf
import random
import os
from tichu.Util import num2action
from tichu.Util import action2num
from tichu.Util import get_available_action_array

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'terminal'])

class Memory(object):

    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, terminal):
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, terminal)
        self.memory.append(transition)

    def sample(self):
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))


class Estimator():

    def __init__(self, scope='estimator', action_num=14, learning_rate=0.001, state_shape=None, mlp_layers=None):
        self.scope = scope
        self.action_num = action_num
        self.learning_rate = learning_rate
        self.state_shape = state_shape if isinstance(state_shape, list) else [state_shape]
        self.mlp_layers = map(int, mlp_layers)

        with tf.compat.v1.variable_scope(scope):
            self._build_model()
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, scope=tf.compat.v1.get_variable_scope().name)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, name='dqn_adam')

        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.compat.v1.train.get_global_step())

    def _build_model(self):
        ### input
        input_shape = [None]
        input_shape.extend(self.state_shape)
        self.X_pl = tf.compat.v1.placeholder(shape=input_shape, dtype=tf.float32, name="X")
        
        ### target value
        self.y_pl = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32, name="y")

        ### action
        self.actions_pl = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32, name="actions")

        ### training
        self.is_train = tf.compat.v1.placeholder(tf.bool, name="is_train")

        batch_size = tf.shape(self.X_pl)[0]

        ### batch normalization
        X = tf.compat.v1.layers.batch_normalization(self.X_pl, training=self.is_train)

        ### Fully connected layers
        fc = tf.contrib.layers.flatten(X)
        for dim in self.mlp_layers:
            fc = tf.contrib.layers.fully_connected(fc, dim, activation_fn=tf.tanh)
#            fc = tf.contrib.layers.fully_connected(fc, dim, activation_fn=tf.nn.elu)
        self.predictions = tf.contrib.layers.fully_connected(fc, self.action_num, activation_fn=None)

        ### Get predictions for the chosn actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        ### Calculate the loss
        self.losses = tf.math.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

    def predict(self, sess, s):
        return sess.run(self.predictions, {self.X_pl: s, self.is_train:False})

    def update(self, sess, s, a, y):
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a, self.is_train: True}
#        _, _, loss = sess.run([tf.compat.v1.train.get_global_step(), self.train_op, self.loss],feed_dict)
        _, loss = sess.run([self.train_op, self.loss],feed_dict)
        return loss


class DQNAgent(object):

    def __init__(self,
                 sess,
                 scope,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor = 0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 action_num=13,
                 state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 learning_rate=0.00005):

        self.use_raw = False
        self.sess = sess
        self.scope = scope
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.action_num = action_num
        self.state_shape = state_shape
        self.train_every = train_every

        self.total_t = 0
        self.train_t = 0
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        self.q_estimator = Estimator(scope=self.scope+"_q", action_num=action_num, learning_rate=learning_rate, state_shape=state_shape, mlp_layers=mlp_layers)
        self.target_estimator = Estimator(scope=self.scope+"_target_q", action_num=action_num, learning_rate=learning_rate, state_shape=state_shape, mlp_layers=mlp_layers)
        
        self.memory = Memory(replay_memory_size, batch_size)
        self.saver = tf.train.Saver()

    def feed(self, ts):
        (state, action, reward, next_state, terminal) = tuple(ts)
        state_input = self.state_parse(state)
        next_state_input = self.state_parse(next_state)
        action_input = action2num(action, state['hand'])
        self.feed_memory(state_input, action_input, reward, next_state_input, terminal)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp>=0 and tmp%self.train_every == 0:
            self.train()

    def step(self, state):
        state_input = self.state_parse(state)
        A = self.predict(state_input)
#        print(A)
        legal_A = get_available_action_array(state['legal_actions'], state['hand'])
#        print(legal_A)
        A = self.remove_illegal(A, legal_A)
#        print(A)
        action = np.random.choice(np.arange(len(A)), p=A)
        action = num2action(action, state['hand'].cards)
        return action

    def eval_step(self, state):
        state_input = self.state_parse(state)
        q_values = self.q_estimator.predict(self.sess, np.expand_dims(state_input, 0))[0]
        legal_A = get_available_action_array(state['legal_actions'], state['hand'])
        probs = self.remove_illegal(np.exp(q_values), legal_A)
        best_action = np.argmax(probs)
        best_action = num2action(best_action, state['hand'].cards)
        return best_action, probs

#    def remove_illegal(self, action, hand_num):
#        probs = np.zeros(self.action_num)
#        for i in range(pow(2,hand_num)):
#            probs[i] = action[i]
#        if np.sum(probs) == 0:
#            for i in range(pow(2,hand_num)):
#                probs[i] = 1 / pow(2,hand_num)
#        else:
#            probs /= sum(probs)
#        return probs

    def remove_illegal(self, action, legal_actions):
        probs = np.multiply(action, legal_actions)
#        probs = np.zeros(self.action_num)
#        for i in range(len(legal_actions)):
#            probs[i] = action[i] * legal_actions[i]
        if np.sum(probs) == 0:
            raise ValueError
        else:
            probs /= sum(probs)
        return probs

    def predict(self, state):
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        A = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
        q_values = self.q_estimator.predict(self.sess, np.expand_dims(state, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    def train(self):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample()

        ### Calculate q values and targets
        q_values_next = self.q_estimator.predict(self.sess, next_state_batch)
        best_actions = np.argmax(q_values_next, axis=1)
        q_values_next_target = self.target_estimator.predict(self.sess, next_state_batch)
        target_batch = reward_batch + np.invert(terminal_batch).astype(np.float32) * \
            self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

        ### Perform gradient descent update
        state_batch = np.array(state_batch)
        loss = self.q_estimator.update(self.sess, state_batch, action_batch, target_batch)
        print('\rINFO - Agent {}, step {}, rl-loss: {}'.format(self.scope, self.total_t, loss), end='')

        ### Update the target estimator
        if self.train_t % self.update_target_estimator_every == 0:
            copy_model_parameters(self.sess, self.q_estimator, self.target_estimator)
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1

    def feed_memory(self, state, action, reward, next_state, terminal):
        self.memory.save(state, action, reward, next_state, terminal)

    def copy_params_op(self, global_vars):
        self_vars = tf.contrib.slim.get_variables(scope=self.scope, collection=tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = []
        for v1, v2 in zip(global_vars, self_vars):
            op = v2.assign(v1)
            update_ops.append(op)
        self.sess.run(update_ops)

    def state_parse(self, state):
        hand = state['hand']
        hand_state = np.zeros(26)
        for i in range(hand.size):
            hand_state[2*i] = hand.cards[i].value
            if hand.cards[i].suit == 'Spade':
                hand_state[2*i+1] = 1
            elif hand.cards[i].suit == 'Heart':
                hand_state[2*i+1] = 2
            elif hand.cards[i].suit == 'Dia':
                hand_state[2*i+1] = 3
            elif hand.cards[i].suit == 'Club':
                hand_state[2*i+1] = 4
            else:
                raise ValueError

        ground_state = np.zeros(3)
        ground_type = state['ground'].type
        if ground_type == 'none':
            ground_state[0] = 0
        elif ground_type == 'solo':
            ground_state[0] = 1
        elif ground_type == 'pair':
            ground_state[0] = 2
        elif ground_type == 'triple':
            ground_state[0] = 3
        elif ground_type == 'four':
            ground_state[0] = 4
        elif ground_type == 'full':
            ground_state[0] = 5
        elif ground_type == 'strat':
            ground_state[0] = 6
        elif ground_type == 'strat_flush':
            ground_state[0] = 7
        elif ground_type == 'pair_seq':
            ground_state[0] = 8
        else:
            raise ValueError
        ground_state[1] = state['ground'].value
        ground_state[2] = state['ground'].player_id

        card_state = np.zeros(3)
        card_state[0] = state['card_num'][1]
        card_state[1] = state['card_num'][2]
        card_state[2] = state['card_num'][3]

        rt_state = np.concatenate((hand_state, ground_state, card_state), axis=None)

        return np.reshape(rt_state, self.state_shape)


    def save(self):
        checkpoint_dir = './train_data/DQN_SY_tmp'
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))
        print(" Saved ")

    def load(self):
        checkpoint_dir = './train_data/DQN_SY_tmp'
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))
        print(" Loaded ")

def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)
    
    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)
