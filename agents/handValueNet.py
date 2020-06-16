from collections import namedtuple
import numpy as np
import tensorflow as tf
import random
import os


class Estimator(object):

    def __init__(self, sess, state_size, learning_rate=0.001):
        self.sess = sess
        self.state_size = state_size
        self.lr = learning_rate

        self.states = tf.placeholder(tf.float32, [1, self.state_size])
        self.target = tf.placeholder(tf.float32)

        self.prediction = self.build_network('pred')
        self.train_op = self.build_optimizer()

    def build_network(self, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            h1 = tf.layers.dense(self.states, 25, activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal(mean=0.01))
            h2 = tf.layers.dense(h1, 125, activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal(mean=0.001))
            h3 = tf.layers.dense(h2, 25, activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal(mean=0.001))
            output = tf.layers.dense(h3, 1, activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal(mean=0.001))
            return output

    def build_optimizer(self):
        value = self.prediction
        loss = tf.reduce_mean(tf.square(self.target - value))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return train_op

    def train_network(self, hand, point):
        target = point[0]
        self.sess.run(self.train_op, feed_dict={self.states: hand, self.target: target})

    def predict(self, hand):
        return self.sess.run(self.prediction, feed_dict={self.states: hand})


class HandValueNet(object):

    def __init__(self, sess,learning_rate=0.00005):

        self.use_raw = False
        self.sess = sess

        self.total_t = 0
        self.train_t = 0

        self.estimator = Estimator(sess, 26)
        self.saver = tf.train.Saver()

    def feed(self, hand, point):
        self.total_t += 1
        hand_input = self.hand_parse(hand)
        self.train(hand_input, point)

    def hand_parse(self, hand):
        hand_input = []
        for i in hand.cards:
            suit = i.suit
            if suit == 'Spade':
                suit = 1
            elif suit == 'Heart':
                suit = 2
            elif suit == 'Dia':
                suit = 3
            elif suit == 'Club':
                suit = 4
            else:
                raise ValueError
            value = i.value
            hand_input.append(suit)
            hand_input.append(value)
        return np.reshape(hand_input, [1, 26])

    def train(self, hand, point):
        self.estimator.train_network(hand, point)

        self.train_t += 1

    def predict(self, hand):
        hand_input = self.hand_parse(hand)
        value = self.estimator.predict(hand_input)
        return value

    def step(self, state):
        return random.choice(state['legal_actions'])

    def eval_step(self, state):
        return random.choice(state['legal_actions'])

    def save(self):
        checkpoint_dir = './train_data/handValue'
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))

    def load(self):
        checkpoint_dir = './train_data/handValue'
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))
