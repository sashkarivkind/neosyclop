#deep RL with continuous policy
#receives batch size and action dimension
#implemented in Tensorflow with Keras API
#uses actor-critic method with 2 neural networks
#actor network is a policy network that outputs the probability of taking an action
#critic network is a value network that outputs the value of a state
#uses the actor network to sample an action
#uses the critic network to estimate the value of a state
#uses the actor network to compute the gradient of the policy
#uses the critic network to compute the temporal difference error
#uses the Adam optimizer to update the actor and critic networks
#uses the Huber loss function for the temporal difference error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks

import numpy as np
import time
import os
import shutil
import matplotlib.pyplot as plt
import copy
import gc

class Buffer:
    def __init__(self,
                 buffer_capacity=10000,
                 batch_size=64,
                 num_states=1,
                 num_actions=1,
                 state_reshape_fn=None):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

        self.state_reshape_fn = state_reshape_fn if state_reshape_fn is not None else lambda x: x


    # Takes (s,a,r,s') observation tuple as input
    # def record(self, obs_tuple):
    #     # Set index to zero if buffer_capacity is exceeded,
    #     # replacing old records
    #     index = self.buffer_counter % self.buffer_capacity
    #
    #     self.state_buffer[index] = obs_tuple[0]
    #     self.action_buffer[index] = obs_tuple[1]
    #     self.reward_buffer[index] = obs_tuple[2]
    #     self.next_state_buffer[index] = obs_tuple[3]
    #
    #     self.buffer_counter += 1

    def record(self, obs_batch):
        # Set indices starting from buffer_counter
        indices = np.arange(self.buffer_counter, self.buffer_counter + self.batch_size) % self.buffer_capacity
        self.state_buffer[indices] = obs_batch[0]   #[:, 0]
        self.action_buffer[indices] = obs_batch[1]   #[:, 1]
        self.reward_buffer[indices] = obs_batch[2]   #[:, 2]
        self.next_state_buffer[indices] = obs_batch[3]   #[:, 3]

        self.buffer_counter += len(obs_batch)

    # We compute the loss and update parameters
    def learn(self, actor_model, target_actor, critic_model, target_critic, actor_optimizer, critic_optimizer, gamma, tau):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch, actor_model, target_actor, critic_model, target_critic, actor_optimizer, critic_optimizer, gamma, tau)

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, actor_model, target_actor, critic_model, target_critic, actor_optimizer, critic_optimizer, gamma, tau):
        # Training and updating Actor & Critic networks.

        #shaping the state according to the state shaping function
        state_batch = self.state_reshape_fn(state_batch)
        next_state_batch = self.state_reshape_fn(next_state_batch)

        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)

            # if done:
            #     y = reward_batch
            # else:
            #     y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)

            y = tf.cast(reward_batch,tf.float32) + gamma * target_critic([next_state_batch, target_actions], training=True)
            # y = gamma * target_critic([next_state_batch, target_actions], training=True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables))


def create_target_network(model):
    target_model = tf.keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())
    return target_model

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))
