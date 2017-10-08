import tensorflow as tf
import numpy as np

class PPO(object):
    """
    --------
    @Example
    --------
    ppo = PPO(n_features=3, n_actions=1, hidden_layer_size=[100],
              c_lf=0.0002, a_lf=0.0001, epsilon=0.2, n_a_learn=10,
              n_c_learn=10)
    
    --------
    @Parameter
    --------
    n_features: int
    n_actions: int
    hidder_layer_sizes: list/array/tuple, default [100]
    c_lf: float, default 0.0002, learning rate of critic network
    a_lf: float, default 0.0001, learning rate of actor network
    epsilon: float, default 0.2
    n_a_learn: int, default 10, learning times of actor network on a batch samples
    n_c_learn: int, default 10, learning times of critic network on a batch samples

    --------
    @Method
    --------
    update(self, s, a, r): update actor and critic using a batch sample (s, a, r)
    choose_action(self, s): choose action on a state s
    get_v(self, s): get value on a state s
    """

    def __init__(self, n_features, n_actions, hidden_layer_sizes=[100],
                 c_lf=0.0002, a_lf = 0.0001, epsilon=0.2, n_a_learn=10, n_c_learn=10):
        self.n_features = n_features
        self.n_actions = n_actions
        self.epsilon = epsilon

        self.hls = hidden_layer_sizes[:]

        self.c_lf = c_lf
        self.a_lf = a_lf

        self.n_a_learn = n_a_learn
        self.n_c_learn = n_c_learn
    
        self._build_actor_critic()

    # actor-critic model
    def _build_actor_critic(self):
        """build actor and critic networks"""
        self.s = tf.placeholder(tf.float32, [None, self.n_features], 'state')
        self.a = tf.placeholder(tf.float32, [None, self.n_actions], 'action')
        self.discounted_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.adv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.device('/gpu:0'):
            self._build_cnet('critic', hidden_layer_sizes=self.hls)
        with tf.device('/gpu:1'):
            pi, pi_params = self._build_anet('actor_pi', trainable=True, 
                                             hidden_layer_sizes=self.hls)
            old_pi, old_pi_params = self._build_anet('actor_old_pi', trainable=False,
                                                     hidden_layer_sizes=self.hls)

            # sample action
            with tf.variable_scope('sample_action'):
                self.sample_op = tf.squeeze(pi.sample(1), axis=0)
            # update old pi
            with tf.variable_scope('update_old_pi'):
                self.update_old_pi_op = [old_p.assign(p) for p, old_p in zip(pi_params, old_pi_params)]

            # actor loss
            with tf.variable_scope('actor_loss'):
                with tf.variable_scope('surrogate'):
                    ratio = pi.prob(self.a) / old_pi.prob(self.a)
                    self.a_loss = -tf.reduce_mean(tf.minimum(
                        ratio * self.adv,
                        tf.clip_by_value(ratio, 1.-self.epsilon, 1.+self.epsilon) * self.adv))
            # actor training
            with tf.variable_scope('actor_train'):
                self.a_train_op = tf.train.AdamOptimizer(self.a_lf).minimize(self.a_loss)#.RMSPropOptimizer(self.a_lf).minimize(self.a_loss)

        config = tf.ConfigProto(allow_soft_placement=True)#, log_device_placement=True
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # graph variables initialized
        self.sess.run(tf.global_variables_initializer())



    # critic network
    def _build_cnet(self, name, hidden_layer_sizes=[100]):
        """build critic, default MLP"""
        with tf.variable_scope(name):
            for i,n_units in enumerate(hidden_layer_sizes):
                if i == 0:
                    c_layer = tf.layers.dense(self.s, n_units, tf.nn.relu)
                else:
                    c_layer = tf.layers.dense(c_layer, n_units,
                                             tf.nn.relu)
            # predict values
            self.v = tf.layers.dense(c_layer, 1)
            self.advantage = self.discounted_r - self.v
            self.c_loss = tf.reduce_mean(tf.square(self.advantage))
            self.c_train_op = tf.train.AdamOptimizer(self.c_lf).minimize(self.c_loss)#.RMSPropOptimizer(self.c_lf).minimize(self.c_loss) 

    # actor network
    def _build_anet(self, name, trainable, hidden_layer_sizes=[100]):
        """build actor(old), default MLP"""
        with tf.variable_scope(name):
            # create hidden layers
            for i,n_units in enumerate(hidden_layer_sizes):
                if i == 0:
                    a_layer = tf.layers.dense(self.s, n_units, tf.nn.relu, trainable=trainable)
                else:
                    a_layer = tf.layers.dense(a_layer, n_units, 
                                             tf.nn.relu, trainable=trainable)
            # action probability distribution
            mu = 2 * tf.layers.dense(a_layer, self.n_actions, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(a_layer, self.n_actions, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    # update
    def update(self, s, a, r):
        self.sess.run(self.update_old_pi_op)
        adv = self.sess.run(self.advantage, {self.s:s, self.discounted_r:r})
        # actor learn
        [self.sess.run(self.a_train_op, {self.s:s, self.a:a, self.adv:adv}) for _ in range(self.n_a_learn)]
        # critic learn
        [self.sess.run(self.c_train_op, {self.s:s, self.discounted_r:r}) for _ in range(self.n_c_learn)]
    
    # choose action
    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.s:s})[0]
        return np.clip(a, -2, 2)
    
    # get value
    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.s:s})[0, 0]

