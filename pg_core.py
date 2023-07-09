import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam

class GaussianPolicyAgent:
    def __init__(self , lr=1e-3, std_deviation=None, model=None, action_dim=2, batch_size=64):
        self.lr = lr
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.std_deviation = std_deviation  # will be None if learning
        if model is None:
            raise ValueError("Must provide a model, default model not implemented yet")
            #self.model = self.build_network()
        else:
            self.model = model

    # def build_network(self):
    #     model = Sequential([
    #         Dense(64, input_dim=self.n_dims, activation='relu'),
    #         Dense(32, activation='relu'),
    #         Dense(4),  # Predicts the mean (2D) and std_dev (2D) of the action
    #     ])
    #     model.compile(loss="mse", optimizer=Adam(lr=self.lr))  # loss function doesn't matter here
    #     return model

    def get_action(self, states, return_stats=False):
        outputs = self.model.predict(states)
        means = outputs[:, :self.action_dim]  # first two values are means
        if self.std_deviation is None:  # learn std_dev
            std_devs = tf.nn.softplus(outputs[:, self.action_dim:])  # last two values are std_dev
        else:
            std_devs = np.full_like(means, self.std_deviation)  # fixed std_dev

        actions = np.random.normal(loc=means, scale=std_devs)
        if return_stats:
            return actions, means, std_devs
        else:
            return actions

    def train(self, states, actions, rewards, masks=None, gamma=0.999, shaper_fn=lambda x: x, max_iterations=1,loud=False):
        # states = states.reshape(-1, self.n_dims)
        #flattens the leading two dimensions of states the remaining dimensions are preserved

        discounted_returns = self.compute_discounted_returns(rewards, masks = masks,gamma=gamma)
        if masks is None:
            masks = np.ones_like(rewards, dtype=np.bool)
        masks = masks.flatten()
        discounted_returns = discounted_returns.flatten()[masks]
        states = states.reshape(-1, *states.shape[2:])[masks]
        actions = actions.reshape(-1, self.action_dim)[masks]

        #create a loop to iterate over the states, actions, and discounted returns
        #at each iteration pick a batch of states, actions, and discounted returns
        #and train the model on that batch
        idx = np.arange(len(states))

        #rundom shuffle the indices
        np.random.shuffle(idx)
        iterations = np.min([max_iterations, len(idx)//self.batch_size])
        if loud:
            print("Training for {} iterations with batchsize {}".format(iterations, self.batch_size))
        for i in range(iterations):
            idx_ = idx[i*self.batch_size:(i+1)*self.batch_size]
            self.train_step(states[idx_], actions[idx_], discounted_returns[idx_], shaper_fn)

    def train_step(self, states, actions, discounted_returns, shaper_fn):

        with tf.GradientTape() as tape:
            outputs = self.model(shaper_fn(states), training=True)
            means = outputs[:, :self.action_dim]
            if self.std_deviation is None:  # learn std_dev
                std_devs = tf.nn.softplus(outputs[:, self.action_dim:])
            else:
                std_devs = np.full_like(means, self.std_deviation)  # fixed std_dev

            loss = self.compute_loss(means, std_devs, actions, discounted_returns)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def compute_loss(self, means, std_devs, actions, discounted_returns):
        neg_log_prob = 0.5 * tf.reduce_sum(tf.square((actions - means) / std_devs), axis=1)
        loss = tf.reduce_mean(neg_log_prob * discounted_returns)
        return loss

    def compute_discounted_returns(self, rewards, masks=None, gamma=0.999):
        #rewards are a numpy array of shape (n_steps, n_envs)
        #returns are a numpy array of shape (n_steps, n_envs)
        #masks are a numpy array of shape (n_steps, n_envs)
        discounting = gamma ** np.arange(len(rewards))
        returns = rewards * discounting[:, None]
        if masks is None:
            returns = returns[::-1].cumsum(axis=0)[::-1]
        else:
            returns = (returns[::-1]*masks[::-1]).cumsum(axis=0)[::-1]
        centered_returns = returns - returns.mean(axis=0)
        return centered_returns





class GaussianPolicyAgent_oldV1:
    def __init__(self , lr=1e-3, std_deviation=None, model=None, action_dim=2, batch_size=64):
        self.lr = lr
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.std_deviation = std_deviation  # will be None if learning
        if model is None:
            raise ValueError("Must provide a model, default model not implemented yet")
            #self.model = self.build_network()
        else:
            self.model = model

    # def build_network(self):
    #     model = Sequential([
    #         Dense(64, input_dim=self.n_dims, activation='relu'),
    #         Dense(32, activation='relu'),
    #         Dense(4),  # Predicts the mean (2D) and std_dev (2D) of the action
    #     ])
    #     model.compile(loss="mse", optimizer=Adam(lr=self.lr))  # loss function doesn't matter here
    #     return model

    def get_action(self, states, return_stats=False):
        outputs = self.model.predict(states)
        means = outputs[:, :self.action_dim]  # first two values are means
        if self.std_deviation is None:  # learn std_dev
            std_devs = tf.nn.softplus(outputs[:, self.action_dim:])  # last two values are std_dev
        else:
            std_devs = np.full_like(means, self.std_deviation)  # fixed std_dev

        actions = np.random.normal(loc=means, scale=std_devs)
        if return_stats:
            return actions, means, std_devs
        else:
            return actions

    #version with timers
    # def get_action(self, states, return_stats=False):
    #     # Timer for predicting outputs
    #     start_time = time.time()
    #     outputs = self.model.predict(states)
    #     print("Time for predicting outputs:", time.time() - start_time)
    #
    #     # Timer for calculating means
    #     start_time = time.time()
    #     means = outputs[:, :self.action_dim]  # first two values are means
    #     print("Time for calculating means:", time.time() - start_time)
    #
    #     # Timer for calculating std_devs
    #     start_time = time.time()
    #     if self.std_deviation is None:  # learn std_dev
    #         std_devs = tf.nn.softplus(outputs[:, self.action_dim:])  # last two values are std_dev
    #     else:
    #         std_devs = np.full_like(means, self.std_deviation)  # fixed std_dev
    #     print("Time for calculating std_devs:", time.time() - start_time)
    #
    #     # Timer for generating random actions
    #     start_time = time.time()
    #     actions = np.random.normal(loc=means, scale=std_devs)
    #     print("Time for generating random actions:", time.time() - start_time)
    #
    #     # Timer for returning actions
    #     start_time = time.time()
    #     if return_stats:
    #         return actions, means, std_devs
    #     else:
    #         return actions

    def train(self, states, actions, rewards, shaper_fn=lambda x: x):
        # states = states.reshape(-1, self.n_dims)
        #flattens the leading two dimensions of states the remaining dimensions are preserved
        states = states.reshape(-1, *states.shape[2:])
        actions = actions.reshape(-1, self.action_dim)
        rewards = rewards.flatten()

        discounted_rewards = self.compute_discounted_rewards(rewards)

        with tf.GradientTape() as tape:
            outputs = self.model(shaper_fn(states), training=True)
            means = outputs[:, :self.action_dim]
            if self.std_deviation is None:  # learn std_dev
                std_devs = tf.nn.softplus(outputs[:, self.action_dim:])
            else:
                std_devs = np.full_like(means, self.std_deviation)  # fixed std_dev

            loss = self.compute_loss(means, std_devs, actions, discounted_rewards)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


    def compute_loss(self, means, std_devs, actions, discounted_rewards):
        neg_log_prob = 0.5 * tf.reduce_sum(tf.square((actions - means) / std_devs), axis=1)
        loss = tf.reduce_mean(neg_log_prob * discounted_rewards)
        return loss


    def compute_discounted_rewards(self, rewards, gamma=0.99): #rewards are a numpy array time x batch
        r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
        r = r[::-1].cumsum()[::-1]
        return r - r.mean()
