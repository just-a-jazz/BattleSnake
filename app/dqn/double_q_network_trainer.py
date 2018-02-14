from trainer import Trainer
from network import Network
from experience_buffer import ExperienceBuffer
import config

import tensorflow as tf
import numpy as np

class DoubleQNetworkTrainer(Trainer):
    def __init__(self,
                 network,
                 sess,
                 enabled=config.is_training,
                 scope=config.dir_name,
                 discount_factor=config.discount_factor,
                 batch_size=config.batch_size,
                 max_buffer_size=config.max_buffer_size,
                 tau=config.tau):
        super(DoubleQNetworkTrainer, self).__init__(discount_factor)
        self._network = network
        self.sess = sess
        self._batch_size = batch_size
        self._tau = tau

        if enabled:
            self._target_network = Network(network.shape, sess, load_model=False)
            self._target_ops = self._update_target_graph(tf.trainable_variables(scope=scope))

        self._experience_buffer = ExperienceBuffer(max_buffer_size)

    def _update_target_graph(self, tf_vars):
        total_vars = len(tf_vars)
        op_holder = []
        for (idx, var) in enumerate(tf_vars[0:total_vars // 2]):
            op_holder.append(tf_vars[idx + total_vars//2].assign(
                (var.value() * self._tau) + ((1-self._tau) * tf_vars[idx + total_vars//2].value())))

        return op_holder

    def _update_target(self):
        for op in self._target_ops:
            self.sess.run(op)

    def train_network(self, state, action, reward, next_state, is_done):
        self._experience_buffer.add(
            np.reshape(np.array([state, action, reward, next_state, is_done]), [1, 5]))

        if len(self._experience_buffer) >= self._batch_size:
            train_batch = self._experience_buffer.sample(self._batch_size)

            '''
            Q1 is the best action for next state predicted from main network
            Q2 is all the actions for next state predicted from target network
            DoubleQ is the value of Q1 from Q2
            targetQ is the output from the neural network for the previous features improved
                by changing the Q using DoubleQ's value
            end_multiplier ensures that if an action caused the episode to end, then its Q
                value is only affected by the reward and not doubleQ
            '''

            Q1 = self.sess.run(self._network.predict,
                               feed_dict={self._network.layers[0]: np.vstack(train_batch[:, 3])})
            Q2 = self.sess.run(self._target_network.layers[-1],
                               feed_dict={self._target_network.layers[0]: np.vstack(train_batch[:, 3])})

            end_multiplier = 1 - train_batch[:, 4]

            double_q = Q2[range(self._batch_size), Q1]
            target_q = train_batch[:,2] + (self._discount_factor * double_q * end_multiplier)

            # Update the network with our target values
            _ = self.sess.run(self._network.train_step,
                              feed_dict={self._network.layers[0]: np.vstack(train_batch[:,0]),
                                         self._network.target_q: target_q,
                                         self._network.actions: train_batch[:,1]})

            self._update_target()
