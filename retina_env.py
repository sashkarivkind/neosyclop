import tensorflow as tf
import numpy as np
from misc import update_with_defaults
import types


class RetinaEnv(object):
    def __init__(self,config,image_generator=None):
        #Set config. Parameters that are not set in config are set to default values
        config = update_with_defaults(default_params=self.load_defaults(),user_params=config)

        self.config = config
        if config.loud:
            print("config: ",config)
        self.foveate = config.foveate
        self.image_h = config.image_h
        self.image_w = config.image_w
        self.sum_axes = config.sum_axes
        self.image_generator = image_generator
        self.filter = config.filter.reshape((1,1,config.history_length,1))
        self.batch_of_ones = tf.ones((config.batch_size,1))

        #set size of observation elements
        self.coordinates_size = 2

        self.retinal_view_size = config.image_h*config.image_w
        self.spectral_density_size = (config.max_freq-config.min_freq)*(1 if self.sum_axes else 2)
        self.location_history_size = self.coordinates_size * self.config.history_length

        self.timestep_size = 1

        #set size of observation
        self.observation_size = self.retinal_view_size + self.spectral_density_size + self.location_history_size + self.timestep_size

        #set size of action
        self.action_size = 2
        self.action_upper_bound = self.config.action_upper_bound
        self.action_lower_bound = -config.action_upper_bound


        #set observation and action spaces is gym-like format
        # self.observation_space = gym.spaces.Box(low=0,high=1,shape=(self.observation_size,))
        # self.action_space = gym.spaces.Box(low=-1,high=1,shape=(self.action_size,))
        # self.observation_space.shape[0]
        # num_actions = env.action_space.shape[0]
        #
        # upper_bound = env.action_space.high[0]
        # lower_bound = env.action_space.low[0]

        #todo: this assingment is currently needed to make the code run
        if image_generator is not None:
            self._generator_ = self.generate_images()

    #method to yield a batch of images upon request
    def generate_images(self):
        for images,_ in self.image_generator:
            yield images

    #receives batch of images
    #converts to grayscale if necessary
    #history_length is the number of images to keep in memory
    #image_h is the height of the image after cropping
    #image_w is the width of the image after cropping
    #image_hm and image_wm are the height and width of the image before cropping
    #initializes tf tensor images_hisory of zeros of size batch_size x image_h x image_w x history_length
    def reset(self, images=None):
        if images is None:
            images = next(self._generator_)
        if self.config.do_grayscale:
            images = rgb2gray(images)
        self.images = images
        self.images_history = tf.zeros((self.config.batch_size,self.config.image_h,self.config.image_w,self.config.history_length))
        self.location_history = tf.zeros((self.config.batch_size,2,self.config.history_length))
        self.timestep = 0
        self.warmup_done = False
        self.cumulative_spectral_density = np.zeros((self.config.batch_size,
                                                     self.spectral_density_size))

        #location is the center of the retinal view
        self.location = tf.zeros((self.config.batch_size,2))
        self.velocity = tf.zeros((self.config.batch_size,2))
        self.acceleration = tf.zeros((self.config.batch_size,2))

        self.observation = np.zeros((self.config.batch_size,self.observation_size))

        return self.observation

    def step(self, action):
        # current state updates
        self.timestep += 1

        if self.config.foveate:
            images = foveate(images, self.config)

        if self.config.motion_mode == 'location':
            self.location = action
        elif self.config.motion_mode == 'velocity':
            self.velocity = action
            self.location += self.velocity
        elif self.config.motion_mode == 'acceleration':
            self.acceleration = action
            self.velocity += self.acceleration
            self.location += self.velocity
        else:
            raise ValueError('motion_mode must be location, velocity or acceleration')

        #round location to nearest integer
        coordinates = tf.cast(tf.round(self.location), tf.int32) #added 0 to avoid round in place
        #clip coordinates to plus minus margin
        coordinates = tf.clip_by_value(coordinates, -self.config.margin, self.config.margin)
        #shift the coordinates to the center of the image, including the margin
        coordinates += self.config.image_hm // 2, self.config.image_wm // 2
        # crop images
        images = crop(self.images, center=coordinates, crop_h=self.config.image_h, crop_w=self.config.image_w)
        self.images_history = tf.concat([self.images_history[:, :, :, 1:], images], axis=3)
        self.location_history = tf.concat([self.location_history[:, :, 1:], self.location[:,:,np.newaxis]], axis=2)
        # retinal view is convolution of images_history with filter
        self.retinal_view = tf.nn.conv2d(self.images_history, self.filter, strides=[1, 1, 1, 1], padding='SAME')

        # compute spectral density
        self.warmup_done = self.timestep > self.config.t_ignore
        if self.warmup_done:
            raw_power = spectral_power_half(self.retinal_view, sum_axes=self.sum_axes)
            if self.sum_axes:
                this_spectral_density = raw_power[:,
                                         self.config.min_freq:self.config.max_freq]
            else:
                this_spectral_density = np.concatenate([x[:,self.config.min_freq:self.config.max_freq] for x in raw_power],
                                                         axis=1)
                # print('this_spectral_density',this_spectral_density.shape)
            #normalize
            if self.config.normalize_spectral_density_per_step:
                this_spectral_density /= tf.reduce_sum(this_spectral_density,axis=1,keepdims=True)
            else: #use predefined normalization
                this_spectral_density /= self.config.fixed_spectral_density_normalization
            #update cumulative_spectral_density
            a = self.timestep - self.config.t_ignore
            self.cumulative_spectral_density = (self.cumulative_spectral_density * (a - 1) + this_spectral_density) / a

        # update observation
        self.observation = self.flatten_observation(self.retinal_view,
                            self.cumulative_spectral_density,
                            self.location_history,
                            self.timestep*self.batch_of_ones)

        # calculate reward
        reward = self.calculate_reward()  # Calculate the reward

        done = self.check_done()  # Check if the episode has ended

        info = {}  # Any additional information you want to provide

        return self.observation, reward, done, info

    def calculate_reward(self):
        #computing mean and variance of cumulative_spectral_density
        if self.warmup_done:
            mean = np.mean(self.cumulative_spectral_density,axis=1,keepdims=True) #keepdims=True is needed to enable broadcasting
            variance = np.mean(tf.square(self.cumulative_spectral_density-mean),axis=1)
            #computing reward as coefficient of variation
            reward = - np.sqrt(variance) / np.squeeze(mean) #here squeeze is needed to remove the extra dimension introduced by keepdims
        else:
            reward = np.zeros((self.config.batch_size))
        #penalize location
        if self.config.distance_penalty_enabled:
            distance = np.sqrt(np.sum(np.square(self.location),axis=1))
            reward -= self.config.distance_penalty_coefficient * np.power(distance/self.config.distance_penalty_r0,self.config.distance_penalty_exponent)
        return reward #tf.cast(reward,tf.float32)

    def check_done(self):
        done = self.timestep >= self.config.t_max
        return done

    def flatten_observation(self, retinal_view, cumulative_spectral_density, location_history, timestep):
        #all tensors are cast to float32 before concatenation
        retinal_view = tf.cast(tf.reshape(retinal_view, [self.config.batch_size, self.retinal_view_size]),tf.float32)
        cumulative_spectral_density = tf.cast(tf.reshape(cumulative_spectral_density, [self.config.batch_size, self.spectral_density_size]),tf.float32)
        location_history = tf.cast(tf.reshape(location_history, [self.config.batch_size, self.coordinates_size*self.config.history_length]),tf.float32)
        timestep = tf.cast(tf.reshape(timestep, [self.config.batch_size, self.timestep_size]),tf.float32)
        observation = tf.concat([retinal_view, cumulative_spectral_density, location_history, timestep], axis=1)
        return observation


    # def unflatten_observation(self, observation):
    #     retinal_view = tf.reshape(observation[:, :self.config.retinal_view_size], [self.config.batch_size, self.config.image_h, self.config.image_w])  # self.config.history_length])
    #     cumulative_spectral_density = tf.reshape(observation[:, self.config.retinal_view_size:self.config.retinal_view_size+self.config.spectral_density_size], [self.config.batch_size, self.config.max_freq-self.config.min_freq])
    #     location_history = tf.reshape(observation[:, self.config.retinal_view_size+self.config.spectral_density_size:self.config.retinal_view_size+self.config.spectral_density_size+self.config.location_history_size], [self.config.batch_size, 2, self.config.history_length])
    #     timestep = tf.reshape(observation[:, -1], [self.config.batch_size, 1])
    #     return retinal_view, cumulative_spectral_density, location_history, timestep

    def unflatten_observation(self,observation):
        retinal_view = tf.reshape(observation[:,:self.retinal_view_size],[self.config.batch_size,self.config.image_h,self.config.image_w])
        cumulative_spectral_density = tf.reshape(observation[:,self.retinal_view_size:self.retinal_view_size+self.spectral_density_size],[self.config.batch_size,self.spectral_density_size])
        location_history = tf.reshape(observation[:,self.retinal_view_size+self.spectral_density_size:self.retinal_view_size+self.spectral_density_size+self.coordinates_size*self.config.history_length],
                                      [self.config.batch_size,self.coordinates_size,self.config.history_length])
        timestep = tf.reshape(observation[:,-1],[self.config.batch_size,1])
        return retinal_view,cumulative_spectral_density,location_history,timestep

    #auxillary version of unflatten_observation that does not depend on batch_size
    def unflatten_observation_v2(self,observation):
        retinal_view = tf.reshape(observation[:,:self.retinal_view_size],
                                  [-1,self.config.image_h,self.config.image_w])
        cumulative_spectral_density = tf.reshape(observation[:,self.retinal_view_size:self.retinal_view_size+self.spectral_density_size],
                                                 [-1,self.spectral_density_size])
        location_history = tf.reshape(observation[:,self.retinal_view_size+self.spectral_density_size:self.retinal_view_size+self.spectral_density_size+self.coordinates_size*self.config.history_length],
                                      [-1,self.coordinates_size,self.config.history_length])
        timestep = tf.reshape(observation[:,-1],[-1,1])
        return retinal_view,cumulative_spectral_density,location_history,timestep

    def load_defaults(self):
        defaults = types.SimpleNamespace()
        defaults.normalize_spectral_density_per_step = True
        defaults.fixed_spectral_density_normalization = 1.0e6 # only used if normalize_spectral_density_per_step is False
        defaults.loud = True
        defaults.sum_axes = True

        defaults.distance_penalty_enabled = False
        defaults.distance_penalty_coefficient = 0.1
        defaults.distance_penalty_exponent = 2
        defaults.distance_penalty_r0 = 15.0

        return defaults



#function to crop images
#images is a batch of images
#center is a batch of coordinates
#crop_h is the height of the cropped image
#crop_w is the width of the cropped image
#returns a batch of cropped images
def crop(images,center,crop_h,crop_w):
    #calculate top left corner of crop
    crop_h2 = crop_h//2
    crop_w2 = crop_w//2
    top_left = tf.cast(tf.stack([center[:,0]-crop_h2,center[:,1]-crop_w2],axis=1),tf.int32)
    #calculate bottom right corner of crop
    bottom_right = tf.cast(tf.stack([center[:,0]+crop_h2,center[:,1]+crop_w2],axis=1),tf.int32)
    #crop images
    images_cropped = tf.map_fn(lambda x: tf.image.crop_to_bounding_box(x[0],x[1][0],x[1][1],crop_h,crop_w),
                               (images, top_left),
                               dtype=tf.float32)
    return images_cropped

#function to convert rgb images to grayscale
#images is a batch of rgb images
#returns a batch of grayscale images
def rgb2gray(images):
    return tf.image.rgb_to_grayscale(images)

#test loop that generates batch of offset coordinates
#and updates the current batch of coordinates
def test():
    config = Config()
    env = RetinaEnv(config)
    env.reset(np.zeros((config.batch_size,config.image_hm,config.image_wm,3)))
    for i in range(10):
        coordinates = np.random.randint(0,config.image_hm,(config.batch_size,2))
        env.step(coordinates)
        print(env.images_history.shape)
        print(env.retinal_view.shape)

#function that receives a batch of images
#computes their spectral power along horizontal and vertical axes,
#and then summing both directions
#fft is computed using numpy
#returns a batch of 1D power spectra
def spectral_power(images, convert_to_grayscale=False,sum_axes=True):
    #convert images to grayscale
    if convert_to_grayscale:
        images = rgb2gray(images)
    #squeeze singular dimensions
    images = np.squeeze(images)
    #compute fft
    fft = np.fft.fft2(images)
    #compute power spectra
    power = np.abs(fft)**2
    #compute power spectra along horizontal axis
    power_h = np.sum(power,axis=1)
    #compute power spectra along vertical axis
    power_v = np.sum(power,axis=2)
    #sum power spectra along both directions
    power_sum = power_h + power_v
    if sum_axes:
        return power_sum
    else:
        return power_h,power_v

#function that returns first half of power spectra
#images is a batch of images
#convert_to_grayscale is a boolean that indicates whether to convert images to grayscale
#returns a batch of 1D power spectra
def spectral_power_half(images, convert_to_grayscale=False,sum_axes=True):
    power = spectral_power(images, convert_to_grayscale=convert_to_grayscale,sum_axes=sum_axes)
    if sum_axes:
        power_half = power[:,:power.shape[1]//2]
    else:
        power_half = power[0][:,:power[0].shape[1]//2],power[1][:,:power[1].shape[1]//2]
    return power_half

def calculate_retinal_filter(t, T1=5, T2=15, n=3, R=0.8):
    # Filter calculation
    t_filter = (t ** n / T1 ** (n + 1)) * np.exp(-t / T1) - R * (t ** n / T2 ** (n + 1)) * np.exp(-t / T2)
    t_filter = t_filter / np.max(t_filter)
    return t_filter