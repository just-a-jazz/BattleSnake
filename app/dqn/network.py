import config

import tensorflow as tf
import numpy as np

class Network(object):
    def __init__(self,
                 shape,
                 sess,
                 scope=config.dir_name,
                 load_model=config.load_model,
                 learning_rate=config.learning_rate,
                 epsilon=config.epsilon,
                 epsilon_min=config.epsilon_min,
                 epsilon_decay=config.epsilon_decay):
        self._shape = shape
        self.sess = sess
        self._create_network(learning_rate, scope)

        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

        self._saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))

        if load_model:
            self.load_network()
        else:
            self.sess.run(tf.global_variables_initializer())

    def _create_network(self, learning_rate, scope):
        assert (len(self._shape) >= 2), 'Invalid shape for network, must have' \
                                       'at least 2 values in array'
        with tf.name_scope(scope):
            # Neural Network Parameters
            num_inputs = self._shape[0]
            num_outputs = self._shape[-1]

            # Neural Network Variables
            weights_and_biases = []
            for i in xrange(0, len(self._shape) - 1):
                weights_and_biases.append((
                    tf.Variable(tf.truncated_normal([self._shape[i], self._shape[i + 1]], stddev=0.1)),
                    tf.Variable(tf.constant(0.1, shape=[self._shape[i + 1]]))))

            # Neural Network Layers
            self.layers = [tf.placeholder(tf.float32, [None, num_inputs])]

            for weights, biases in weights_and_biases[:-1]:
                new_layer = tf.nn.relu(tf.add(tf.matmul(self.layers[-1], weights.initialized_value()),
                                              biases.initialized_value()))
                self.layers.append(new_layer)

            output_layer = tf.add(tf.matmul(self.layers[-1], weights_and_biases[-1][0].initialized_value()),
                                  weights_and_biases[-1][1].initialized_value())
            self.layers.append(output_layer)

            # Predict best action
            self.predict = tf.argmax(self.layers[-1], 1)

            self.target_q = tf.placeholder(tf.float32, [None])
            self.actions = tf.placeholder(tf.int32, [None])
            self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.layers[-1], self.actions_onehot), axis=1)

            # How to improve the neural network
            self.loss = tf.reduce_mean(tf.square(self.target_q - self.Q))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            self.train_step = optimizer.minimize(self.loss)

    def get_output(self, state):
        best_output, o_p = self.sess.run([self.predict, self.layers[-1]],
                                          feed_dict={self.layers[0]: state})

        # Epsilon greedy policy
        if np.random.rand(1) < self._epsilon and config.is_training:
            print('Taking random action')
            output = np.array([np.random.randint(0, 4)])[0]
        else:
            output = best_output

        print(o_p[0])

        # Decay epsilon
        if self._epsilon > self._epsilon_min:
            self._epsilon = self._epsilon * self._epsilon_decay

        return output

    def load_network(self, filepath='dqn/data/' + config.dir_name + '/model.cfk'):
        self._saver.restore(self.sess, filepath)

    def save_network(self, filepath='dqn/data/' + config.dir_name + '/model.cfk'):
        self._saver.save(self.sess, filepath)
