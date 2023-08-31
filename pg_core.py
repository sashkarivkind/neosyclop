import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam

class GaussianPolicyAgent:
    def __init__(self , lr=1e-3, std_deviation=None, model=None, action_dim=2, batch_size=64, do_clip=False, clip=0.2):
        self.lr = lr
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.std_deviation = std_deviation  # will be None if learning
        self.do_clip = do_clip
        self.clip = clip
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

    def get_action(self, states, return_stats=False, return_log_prob=False):
        outputs = self.model.predict(states)
        means = outputs[:, :self.action_dim]  # first two values are means
        if self.std_deviation is None:  # learn std_dev
            std_devs = tf.nn.softplus(outputs[:, self.action_dim:])  # last two values are std_dev
        else:
            std_devs = np.full_like(means, self.std_deviation)  # fixed std_dev

        # actions = np.random.normal(loc=means, scale=std_devs)
        # if return_stats:
        #     return actions, means, std_devs
        # else:
        #     return actions
        actions = np.random.normal(loc=means, scale=std_devs)
        if return_log_prob:
            log_probs = -0.5 * np.sum(np.square((actions - means) / std_devs), axis=1)
            if return_stats:
                return actions, means, std_devs, log_probs
            else:
                return actions, log_probs
        elif return_stats:
            return actions, means, std_devs
        else:
            return actions

    def train(self, states, actions, rewards, log_probs=None, masks=None, gamma=0.999, shaper_fn=lambda x: x, max_iterations=1,loud=False,centered=True):
        # states = states.reshape(-1, self.n_dims)
        #flattens the leading two dimensions of states the remaining dimensions are preserved

        discounted_returns = self.compute_discounted_returns(rewards, masks = masks,gamma=gamma,centered=centered)
        if masks is None:
            masks = np.ones_like(rewards, dtype=np.bool)
        masks = masks.flatten()
        discounted_returns = discounted_returns.flatten()[masks]
        states = states.reshape(-1, *states.shape[2:])[masks]
        actions = actions.reshape(-1, self.action_dim)[masks]
        if log_probs is not None:
            # #assert that the log_probs are the same shape as the actions
            # assert log_probs.shape == actions.shape
            #store the log_probs
            # log_probs = log_probs.reshape(-1, self.action_dim)[masks]
            log_probs = log_probs.reshape(-1)[masks]

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
            self.train_step(states[idx_], actions[idx_], discounted_returns[idx_], shaper_fn, log_probs=log_probs[idx_])

    def train_step(self, states, actions, discounted_returns, shaper_fn, log_probs=None):

        with tf.GradientTape() as tape:
            outputs = self.model(shaper_fn(states), training=True)
            means = outputs[:, :self.action_dim]
            if self.std_deviation is None:  # learn std_dev
                std_devs = tf.nn.softplus(outputs[:, self.action_dim:])
            else:
                std_devs = np.full_like(means, self.std_deviation)  # fixed std_dev

            loss = self.compute_loss(means, std_devs, actions, discounted_returns, old_log_prob=None if not self.do_clip else log_probs)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    # @tf.function
    def compute_loss(self, means, std_devs, actions, discounted_returns, old_log_prob=None):
        if self.do_clip:
            #compute probability of actions under the old policy
            #compute probability of actions under the new policy
            #compute the ratio of the probabilities
            #clip the ratio of the probabilities
            #compute the loss
            #return the loss

            # old_means = tf.stop_gradient(means)
            # old_std_devs = tf.stop_gradient(std_devs)
            # old_log_prob = -0.5 * tf.reduce_sum(tf.square((actions - old_means) / old_std_devs), axis=1)
            new_log_prob = -0.5 * tf.reduce_sum(tf.square((actions - means) / std_devs), axis=1)
            ratio = tf.exp(new_log_prob - old_log_prob)
            clipped_ratio = tf.clip_by_value(ratio, 1-self.clip, 1+self.clip)
            loss = -tf.reduce_mean(tf.minimum(ratio * discounted_returns, clipped_ratio * discounted_returns))
            return loss
        else:
            neg_log_prob = 0.5 * tf.reduce_sum(tf.square((actions - means) / std_devs), axis=1)
            loss = tf.reduce_mean(neg_log_prob * discounted_returns)
            return loss

    def compute_discounted_returns(self, rewards, masks=None, gamma=0.9999, centered=True):
        #rewards are a numpy array of shape (n_steps, n_envs)
        #returns are a numpy array of shape (n_steps, n_envs)
        #masks are a numpy array of shape (n_steps, n_envs)
        discounting = gamma ** np.arange(len(rewards))
        returns = rewards * discounting[:, None]
        if masks is None:
            returns = returns[::-1].cumsum(axis=0)[::-1]
        else:
            returns = (returns[::-1]*masks[::-1]).cumsum(axis=0)[::-1]
        centered_returns = returns - (returns.mean(axis=1,keepdims=True) if centered else 0)
        return centered_returns


#a version of the policy agent that uses a replay buffer
class GaussianPolicyAgentWithBuffer(GaussianPolicyAgent):
    def __init__(self, lr=1e-3, std_deviation=None, model=None,state_dim=0, action_dim=2, batch_size=64, buffer_size=10000):
        super().__init__(lr=lr, std_deviation=std_deviation, model=model, action_dim=action_dim, batch_size=batch_size)
        self.buffer = ReplayBuffer(buffer_size,stored_items_and_dims={'states': state_dim, 'actions': action_dim, 'returns': 1})
        self.state_dim = state_dim
        self.batch_size = batch_size

    def record(self, states, actions, rewards, masks=None, num_records=None, gamma=0.999, centered=True):
        discounted_returns = self.compute_discounted_returns(rewards, masks = masks,gamma=gamma,centered=centered)
        if masks is None:
            masks = np.ones_like(rewards, dtype=np.bool)
        masks = masks.flatten()
        discounted_returns = discounted_returns.flatten()[masks][...,np.newaxis]
        states = states.reshape(-1, *states.shape[2:])[masks]
        actions = actions.reshape(-1, self.action_dim)[masks]
        self.buffer.record({'states': states, 'actions': actions, 'returns': discounted_returns}, num_records=num_records)

    def train(self, shaper_fn=lambda x: x, max_iterations=1, loud=False):
        for i in range(max_iterations):
            batch = self.buffer.sample(self.batch_size)
            self.train_step(batch['states'], batch['actions'], batch['returns'], shaper_fn=shaper_fn)

#buffer class for storing states, actions, and rewards
#only stores states, actions, and rewards that have masks of 1
#each time store is called, the last records are popped off the buffer
class ReplayBuffer:

    def __init__(self,
                 buffer_capacity=10000,
                 stored_items_and_dims={'states': 2, 'actions': 2, 'rewards': 1, 'next_states': 2, 'dones': 1}):

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity

        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        self.buffer = {key: np.zeros([self.buffer_capacity, dims]) for key,dims in stored_items_and_dims.items()}

    def record(self, observation_dict, num_records=None, observation_selector='random'):
        if num_records is None:
            num_records = len(observation_dict[list(observation_dict.keys())[0]])
            observation_indices = np.arange(num_records)
        else:
            if observation_selector == 'random':
                observation_indices = np.random.choice(len(observation_dict[list(observation_dict.keys())[0]]), num_records)
            else:
                raise NotImplementedError
        # print('num_records', num_records)
        # print('observation_indices', observation_indices)
        # Set indices starting from buffer_counter
        indices = np.arange(self.buffer_counter, self.buffer_counter + num_records) % self.buffer_capacity
        for item in observation_dict.keys():
            # print('item', item)
            self.buffer[item][indices] = observation_dict[item][observation_indices]

        self.buffer_counter += num_records

    def sample(self, batch_size=64):
        # Randomly sample indices
        indices = np.random.choice(np.min([self.buffer_capacity,self.buffer_counter]), batch_size)
        # print('indices', indices)
        return {item: self.buffer[item][indices] for item in self.buffer.keys()}

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
