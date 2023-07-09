import numpy as np

class BatchMountainCarContinuous:
    def __init__(self, batch_size, gravity=0.0025, force=0.001, start_position=-0.5,max_steps=300):
        self.gravity = gravity
        self.force = force
        self.start_position = start_position
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.reset()

    def step(self, actions):
        actions = actions.reshape(-1)
        positions, velocities = self.states.T
        velocities += actions*self.force - self.gravity*np.cos(3*positions)
        velocities = np.clip(velocities, -0.07, 0.07)
        positions += velocities
        positions = np.clip(positions, -1.2, 0.6)

        dones = positions >= 0.5

        rewards = np.where(dones, 100.0, -1.0)

        self.states = np.column_stack((positions, velocities))
        self.step_counts += 1

        dones = np.logical_or(dones, self.step_counts >= self.max_steps)

        return self.states, rewards, dones, [{}]*self.batch_size

    def reset(self):
        self.states = np.full((self.batch_size, 2), [self.start_position, 0])
        self.step_counts = np.zeros(self.batch_size, dtype=int)
        return self.states


class SimpleMountainCarContinuous:
    def __init__(self, gravity=0.0025, force=0.001, start_position=-0.5,max_steps=300):
        self.gravity = gravity
        self.force = force
        self.start_position = start_position
        self.max_steps = max_steps
        self.reset()

    def step(self, action):
        position, velocity = self.state

        velocity += action*self.force - self.gravity*np.cos(3*position)
        velocity = np.clip(velocity, -0.07, 0.07)
        position += velocity
        position = np.clip(position, -1.2, 0.6)

        done = bool(position>=0.5)

        reward = -1.0 if not done else 100.0

        self.state = np.array([position, velocity])
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([self.start_position, 0])
        self.step_count = 0
        return self.state


class SimpleCartPole:
    def __init__(self, g=9.8, m_c=1.0, m_p=0.1, l=0.5, dt=0.02, x_threshold=2.4, theta_threshold=12 * 2 * np.pi / 360):
        self.g = g  # gravity
        self.m_c = m_c  # mass of cart
        self.m_p = m_p  # mass of pole
        self.l = l  # length of pole
        self.dt = dt  # time step
        self.x_threshold = x_threshold  # max cart position
        self.theta_threshold = theta_threshold  # max pole angle in radians
        self.reset()

    def step(self, action):
        x, v, theta, omega = self.state

        force = action
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        temp = (force + self.m_p * self.l * omega * omega * sin_theta) / (self.m_c + self.m_p)
        theta_acc = (self.g * sin_theta - cos_theta * temp) / (self.l * (4.0/3.0 - self.m_p * cos_theta * cos_theta / (self.m_c + self.m_p)))
        x_acc = temp - self.m_p * self.l * theta_acc * cos_theta / (self.m_c + self.m_p)

        x += self.dt * v
        v += self.dt * x_acc
        theta += self.dt * omega
        omega += self.dt * theta_acc

        self.state = np.array([x, v, theta, omega])

        done = x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold or theta > self.theta_threshold
        reward = 1.0 if not done else 0.0

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        return self.state



class SimpleCartPole_v2: #simple version of cartpole
    def __init__(self, g=9.8, m_c=1.0, m_p=0.1, l=0.5, dt=0.02, x_threshold=2.4, theta_threshold=12 * 2 * np.pi / 360):
        self.g = g  # gravity
        self.m_c = m_c  # mass of cart
        self.m_p = m_p  # mass of pole
        self.l = l  # length of pole
        self.dt = dt  # time step
        self.x_threshold = x_threshold  # max cart position
        self.theta_threshold = theta_threshold  # max pole angle in radians
        self.reset()

    def step(self, action):
        x, v, theta, omega = self.state

        force = action
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        temp = (force + self.m_p * self.l * omega * omega * sin_theta) / (self.m_c + self.m_p)
        theta_acc = (self.g * sin_theta - cos_theta * temp) / (
                    self.l * (4.0 / 3.0 - self.m_p * cos_theta * cos_theta / (self.m_c + self.m_p)))
        x_acc = temp - self.m_p * self.l * theta_acc * cos_theta / (self.m_c + self.m_p)

        x += self.dt * v
        v += self.dt * x_acc
        theta += self.dt * omega
        omega += self.dt * theta_acc

        self.state = np.array([x, v, theta, omega])

        done = x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold or theta > self.theta_threshold

        # Reward is 1 for every step taken, including the termination step
        reward = 1.0

        # Add penalty for distance from center and for pole angle
        reward -= 0.01 * np.abs(x) + 0.01 * (theta ** 2)

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        return self.state




