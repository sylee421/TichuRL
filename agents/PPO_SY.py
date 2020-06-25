import tensorflow as tf
import numpy as np
from tichu.Util import state_parse, num2action, action2num, get_available_action_array

class PPONetwork(object):

    def __init__(self, sess, obs_dim, act_dim, name):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.name = name

        def ortho_init(scale=1.0):
            def _ortho_init(shape, dtype, partition_info=None):
                shape = tuple(shape)
                if len(shape) == 2:
                    flat_shape = shape
                elif len(shape) == 4:
                    flat_shape = (np.prod(shape[:-1]), shape[-1])
                else:
                    raise NotImplementedError
                a = np.random.normal(0.0, 1.0, flat_shape)
                u, _, v = np.linalg.svd(a, full_matrices=False)
                q = u if u.shape == flat_shape else v
                q = q.reshape(shape)
                return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
            return _ortho_init

        def fc(x, scope, nh, init_scale=1.0, init_bias=0.0):
            with tf.variable_scope(scope):
                nin = x.get_shape()[1].value
                w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
                b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
                return tf.matmul(x, w)+b

        with tf.variable_scope(name):
            X = tf.placeholder(tf.float32, [None, obs_dim], name="input")
            available_moves = tf.placeholder(tf.float32, [None, act_dim], name="availableActions")
            activation = tf.nn.relu
            h1 = activation(fc(X,'fc1',nh=512, init_scale=np.sqrt(2)))
            h2 = activation(fc(h1,'fc2',nh=256, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', act_dim, init_scale = 0.01)
            h3 = activation(fc(h1,'fc3',nh=256, init_scale=np.sqrt(2)))
            vf = fc(h3, 'vf', 1)[:,0]
        availPi = tf.add(pi, available_moves)

        def sample():
            u = tf.random_uniform(tf.shape(availPi))
            t = availPi - tf.log(-tf.log(u))
            return tf.argmax(tf.multiply(t, available_moves), axis=-1)

        a0 = sample()
        el0in = tf.exp(availPi - tf.reduce_max(availPi, axis=-1, keep_dims=True))
        z0in = tf.reduce_sum(el0in, axis=-1, keep_dims=True)
        p0in = el0in / z0in
        onehot = tf.one_hot(a0, availPi.get_shape().as_list()[-1])
        neglogpac = -tf.log(tf.reduce_sum(tf.multiply(p0in, onehot), axis=-1))

        def step(obs, availAcs):
            a, v, neglogp = sess.run([a0, vf, neglogpac], {X:obs, available_moves:availAcs})
            return a, v, neglogp

        def value(obs, availAcs):
            return sess.run(vf, {X:obs, available_moves:availAcs})

        self.availPi = availPi
        self.neglogpac = neglogpac
        self.X = X
        self.available_moves = available_moves
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        def getParams():
            return sess.run(self.params)

        self.getParams = getParams

        def loadParams(paramsToLoad):
            restores = []
            for p, loadedP in zip(self.params, paramsToLoad):
                restores.append(p.assign(loadedP))
            sess.run(restores)

        self.loadParams = loadParams

        def saveParams(path):
            modelParams = sess.run(self.params)
            joblib.dump(modelParams, path)

        self.saveParams = saveParams


class PPOModel(object):

    def __init__(self, sess, network, inpDim, actDim, ent_coef, vf_coef, max_grad_norm):

        self.network = network

        ACTIONS = tf.placeholder(tf.int32, [None], name='actionsPlaceholder')
        ADVANTAGES = tf.placeholder(tf.float32, [None], name='advantagesPlaceholder')
        RETURNS = tf.placeholder(tf.float32, [None], name='returnsPlaceholder')
        OLD_NEG_LOG_PROB_ACTIONS = tf.placeholder(tf.float32, [None], name='oldNegLogProbActionsPlaceholder')
        OLD_VAL_PRED = tf.placeholder(tf.float32, [None], name='oldValPlaceholder')
        LEARNING_RATE = tf.placeholder(tf.float32, [], name='LRplaceholder')
        CLIP_RANGE = tf.placeholder(tf.float32, [], name='cliprangePlaceholder')

        l0 = network.availPi - tf.reduce_max(network.availPi, axis=-1, keep_dims=True)
        el0 = tf.exp(l0)
        z0 = tf.reduce_sum(el0, axis=-1, keep_dims=True)
        p0 = el0 / z0
        entropy = -tf.reduce_sum((p0+1e-8) * tf.log(p0+1e-8), axis=-1)
        oneHotActions = tf.one_hot(ACTIONS, network.pi.get_shape().as_list()[-1])
        neglogpac = -tf.log(tf.reduce_sum(tf.multiply(p0, oneHotActions), axis=-1))

#        def neglogp(state, actions, index):
#            return sess.run(neglogpac, {network.X: state, network.available_moves: actions, ACTIONS: index})
#
#        self.neglogp = neglogp

        entropyLoss = tf.reduce_mean(entropy)

        v_pred = network.vf
        v_pred_clipped = OLD_VAL_PRED + tf.clip_by_value(v_pred - OLD_VAL_PRED, -CLIP_RANGE, CLIP_RANGE)
        vf_losses1 = tf.square(v_pred - RETURNS)
        vf_losses2 = tf.square(v_pred_clipped - RETURNS)
        vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        prob_ratio = tf.exp(OLD_NEG_LOG_PROB_ACTIONS - neglogpac)
        pg_losses1 = -ADVANTAGES * prob_ratio
        pg_losses2 = -ADVANTAGES * tf.clip_by_value(prob_ratio, 1.0-CLIP_RANGE, 1.0+CLIP_RANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))

        loss = pg_loss + vf_coef*vf_loss - ent_coef*entropyLoss

        params = network.params
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, observations, availableActions, returns, actions, values, neglogpacs):
            advs = returns - values
            advs = (advs-advs.mean()) / (advs.std() + 1e-8)
            inputMap = {network.X: observations, network.available_moves: availableActions, ACTIONS: actions, ADVANTAGES: advs, RETURNS: returns, OLD_VAL_PRED: values, OLD_NEG_LOG_PROB_ACTIONS: neglogpacs, LEARNING_RATE: lr, CLIP_RANGE: cliprange}
            return sess.run([pg_loss, vf_loss, entropyLoss, _train], inputMap)[:-1]

        self.train = train


class PPOAgent(object):

    def __init__(self, sess, inpDim=32, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
        
        self.trainingNetwork = PPONetwork(sess, inpDim, 8192, "trainNet")
        self.trainingModel = PPOModel(sess, self.trainingNetwork, inpDim, 8192, ent_coef, vf_coef, max_grad_norm)

        self.inpDim = inpDim

        self.obs_in = []
        self.availAcs_in = []
        self.returns_in = []
        self.actions_in = []
        self.values_in = []
        self.neglogpacs_in = []

    def feed(self, trajectories):

        lrnow = 0.00025
        cliprangenow = 0.2

        for ts in trajectories:
            (state, action, reward, next_state, terminal) = tuple(ts)

            self.obs_in.append(state_parse(state))
            self.availAcs_in.append( get_available_action_array(state['legal_actions'], state['hand']) )
            self.returns_in.append(reward)
            self.actions_in.append(action2num(action, state['hand']))

        self.trainingModel.train(lrnow, cliprangenow, np.array(self.obs_in), np.array(self.availAcs_in), np.array(self.returns_in), np.array(self.actions_in), np.array(self.values_in).reshape(-1), np.array(self.neglogpacs_in).reshape(-1))

        self.obs_in = []
        self.availAcs_in = []
        self.returns_in = []
        self.actions_in = []
        self.values_in = []
        self.neglogpacs_in = []

    def step(self, state):
        obs = state_parse(state)
        obs = np.reshape(obs, [1, self.inpDim])
        availAcs = get_available_action_array(state['legal_actions'], state['hand'])
        availAcs = np.reshape(availAcs, [1, 8192])
        a, v, neglogp = self.trainingNetwork.step(obs, availAcs)
        a = num2action(a, state['hand'].cards)
        self.values_in.append(v)
        self.neglogpacs_in.append(neglogp)
        return a
