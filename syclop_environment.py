#defines an RL environment for syclop
#syclop receives a batch of images and returns a batch of actions
#images are updated by the environment
#actions are updated by the agent

class Scene():
    def __init__(self,image_matrix = None, frame_list = None ):
        if image_matrix is not None:
            self.image = image_matrix
        if frame_list is not None:
            self.movie_mode = True
            self.current_frame = 0
            self.frame_list = frame_list
            self.total_frames = len(self.frame_list)
            self.image = self.frame_list[self.current_frame]
        self.maxy, self.maxx = np.shape(self.image)[:2]
        self.hp = HP()

    def edge_image_x(self,edge_location,contrast=1.0):
        self.image = np.zeros(self.image.shape)
        self.image[:,int(edge_location)] = contrast


class Sensor():
    def __init__(self, log_mode=False, log_floor = 1e-3, fading_mem=0.0, fisheye=None,**kwargs):

        defaults = HP()
        defaults.winx = 16*4
        defaults.winy = 16*4
        defaults.centralwinx = 4*4
        defaults.centralwiny = 4*4
        defaults.fading_mem = fading_mem
        defaults.memorize_polarity = False #not in use
        defaults.fisheye = fisheye
        defaults.resolution_fun = None
        defaults.resolution_fun_type = 'down_and_up'
        defaults.nchannels = None

        self.hp = HP()
        self.log_mode = log_mode
        self.log_floor = log_floor
        self.hp.upadte_with_defaults(att=kwargs, default_att=defaults.__dict__)

        self.frame_size = self.hp.winx * self.hp.winy
        self.reset()



    def reset(self):
        self.frame_view = np.zeros([self.hp.winy,self.hp.winx]+([] if self.hp.nchannels is None else [self.hp.nchannels]))
        if self.hp.resolution_fun_type == 'down':
            self.frame_view = self.hp.resolution_fun(self.frame_view)

        framey=self.frame_view.shape[0]
        framex=self.frame_view.shape[1]
        self.cwx1 = (framex-self.hp.centralwinx)//2
        self.cwy1 = (framey-self.hp.centralwiny)//2
        self.cwx2 = self.cwx1 + self.hp.centralwinx
        self.cwy2 = self.cwy1 + self.hp.centralwiny
        # if self.hp.resolution_fun is not None:
        #     self.frame_view=self.hp.resolution_fun(self.frame_view)
        # print(  'debug1', np.shape(self.frame_view))
        self.central_frame_view = self.frame_view[self.cwy1:self.cwy2,self.cwx1:self.cwx2]
        # print(  'debug2', np.shape(self.central_frame_view))

        self.dvs_view =self.dvs_fun(self.frame_view, self.frame_view)
        self.central_dvs_view =self.dvs_view[self.cwy1:self.cwy2,self.cwx1:self.cwx2]


    def update(self,scene,agent):
        current_view = self.get_view(scene, agent)
        self.dvs_view = self.dvs_view*self.hp.fading_mem+self.dvs_fun(current_view, self.frame_view)
        self.central_dvs_view =self.dvs_view[self.cwy1:self.cwy2,self.cwx1:self.cwx2]
        self.frame_view = current_view
        self.central_frame_view = self.frame_view[self.cwy1:self.cwy2,self.cwx1:self.cwx2]

    def dvs_fun(self,current_frame, previous_frame):
        delta = current_frame - previous_frame
        if self.log_mode:
            return (np.log10(np.abs(delta)+self.log_floor)-np.log(self.log_floor))*np.sign(delta)
        else:
            return current_frame - previous_frame

    def get_view(self,scene,agent):
        if self.hp.fisheye is None:
            img=scene.image
        else:
            cam = 0. + self.hp.fisheye['cam']
            cam[:2,2]=np.array([agent.q[0]+self.hp.winx//2, scene.maxy - agent.q[1] - self.hp.winy//2])
            # print('debug cam', cam)
            # print('debug cv openCL:', cv2.ocl.useOpenCL())
            img = cv2.undistort(scene.image,cam,self.hp.fisheye['dist'])
        view =  img[scene.maxy - agent.q[1] - self.hp.winy: scene.maxy - agent.q[1],
           agent.q[0]: agent.q[0]+self.hp.winx]
        if self.hp.resolution_fun is not None:
            view = self.hp.resolution_fun(view)
        return view

class Agent():
    def __init__(self,max_q = None):
        self.hp = HP()
        # self.hp.action_space =[-1,1]# [-3,-2,-1,0,1,2,3]
        # self.hp.action_space = ['v_right','v_left','v_up','v_down','null'] #'            #,'R','L','U','D'] +
        # self.hp.action_space = ['v_right','v_left','v_up','v_down','null','R','L','U','D'] + \
        #                       [['v_right','v_up'],['v_right','v_down'],['v_left','v_up'],['v_left','v_down']]#'
        self.hp.action_space = ['v_right','v_left','v_up','v_down','null'] + \
                              [['v_right','v_up'],['v_right','v_down'],['v_left','v_up'],['v_left','v_down']]#'
        self.hp.big_move = 25
        self.max_q = max_q
        self.q_centre = np.array(self.max_q, dtype='f') / 2
        self.saccade_flag = False
        self.reset()

    def reset(self, centered=False, q_init=None):
        if q_init is not None:
            self.q_ana=q_init+0.
        elif centered:
            self.q_ana = np.array(self.max_q)/2.
        else:
            self.q_ana = np.array([np.random.randint(self.max_q[0]), np.random.randint(self.max_q[1])], dtype='f')
        self.qdot = np.array([0.0,0.0])
        self.qdotdot = np.array([0.0,0.0])
        self.q = np.int32(np.floor(self.q_ana))

    def set_manual_q(self,q):
        self.q = q

    def set_manual_trajectory(self,manual_q_sequence,manual_t=0):
        self.manual_q_sequence = manual_q_sequence
        self.manual_t = manual_t

    def manual_act(self):
        self.q = self.manual_q_sequence[self.manual_t]
        self.manual_t += 1
        self.manual_t %= len(self.manual_q_sequence)

    def act(self,a):
        if a is None:
            action = 'null'
        else:
            action = self.hp.action_space[a]

        self.saccade_flag = False

        #delta_a = 0.001
        if type(action) == list:
            for subaction in action:
                self.parse_action(subaction)
        else:
            self.parse_action(action)

        #print('debug', self.max_q, self.q_centre)
        self.qdot += self.qdotdot
        #self.qdot -= self.hp.returning_force*(self.q_ana-self.q_centre)
        self.q_ana +=self.qdot
        self.enforce_boundaries()

    def enforce_boundaries(self):
        self.q_ana = np.minimum(self.q_ana,self.max_q)
        self.q_ana = np.maximum(self.q_ana,[0.0, 0.0])
        self.q = np.int32(np.floor(self.q_ana))

    def parse_action(self,action):
        if type(action)==int: #todo  - int actions denote velocity shift of velocity in x direction. this needs to be generalized
            self.qdot[0] = action
            self.qdotdot = np.array([0., 0.])
        elif action == 'reset':
            self.reset()
            self.saccade_flag = True
        elif action == 'R':
            self.q_ana[0] += self.hp.big_move
            self.qdot = np.array([0.0, 0.0])
            self.qdotdot = np.array([0.0, 0.0])
            self.saccade_flag = True
            self.enforce_boundaries()

        elif action == 'L':
            self.q_ana[0] -= self.hp.big_move
            self.qdot = np.array([0.0, 0.0])
            self.qdotdot = np.array([0.0, 0.0])
            self.saccade_flag = True
            self.enforce_boundaries()

        elif action == 'U':
            self.q_ana[1] += self.hp.big_move
            self.qdot = np.array([0.0, 0.0])
            self.qdotdot = np.array([0.0, 0.0])
            self.saccade_flag = True
            self.enforce_boundaries()

        elif action == 'D':
            self.q_ana[1] -= self.hp.big_move
            self.qdot = np.array([0.0, 0.0])
            self.qdotdot = np.array([0.0, 0.0])
            self.saccade_flag = True
            self.enforce_boundaries()

        elif action == 'v_up':  # up
            self.qdot[1] = self.qdot[1] + 1 if self.q[1] < self.max_q[1] - 1 else -0
            self.qdotdot = np.array([0., 0.])
        elif action == 'v_down':   # down
            self.qdot[1] = self.qdot[1] - 1 if self.q[1] > 0 else -0
            self.qdotdot = np.array([0., 0.])
        elif action == 'v_left':   # left
            self.qdot[0] = self.qdot[0]-1 if self.q[0] > 0 else -0
            self.qdotdot = np.array([0., 0.])
        elif action == 'v_right':   # right
            self.qdot[0] = self.qdot[0]+1 if self.q[0] < self.max_q[0]-1 else -0
            self.qdotdot = np.array([0.,0.])
        elif action == 'a_up':   # up
            self.qdotdot[1] = self.qdotdot[1] + delta_a if self.q[1] < self.max_q[1]-1 else -0
        elif action == 'a_down':   # down
            self.qdotdot[1] = self.qdotdot[1] - delta_a if self.q[1] > 0 else -0
        elif action == 'a_left':   # left
            self.qdotdot[0] = self.qdotdot[0] - delta_a if self.q[0] > 0 else -0
        elif action == 'a_right':   # right
            self.qdotdot[0] = self.qdotdot[0] + delta_a if self.q[0] < self.max_q[0]-1 else -0
        elif action == 'null':   # null
            pass
        else:
            error('unknown action')

#function receives a batch of images and a batch of crop centers and returns a batch of cropped images
#by default images are greyscale, but can be RGB if specified
def crop_images(images, centers, cropsize, RGB=False):
    if RGB:
        cropped_images = np.zeros((images.shape[0], cropsize, cropsize, 3))
    else:
        cropped_images = np.zeros((images.shape[0], cropsize, cropsize))

    #ensure that the cropsize is even
    if cropsize % 2 == 1:
        cropsize += 1

    #centers are rounded to the nearest integer
    centers = np.int32(np.round(centers))

    for i in range(images.shape[0]):
        cropped_images[i] = images[i, centers[i, 0] - cropsize // 2:centers[i, 0] + cropsize // 2,
                            centers[i, 1] - cropsize // 2:centers[i, 1] + cropsize // 2]
    return cropped_images


#function that creates a single gabor filter
def gabor_kernel(frequency=1, theta=0, bandwidth=1):
    sigma = 1 / (2 * np.pi * bandwidth)  # bandwidth = 1 / (2 * pi * sigma)
    sigma_x = sigma
    sigma_y = sigma / frequency

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(
        2 * np.pi * frequency * x_theta)
    return gb

#function creates a bank of gabor filters
#and returns Keras 2D convolutional layers with the filters as weights
def create_gabor_bank(hp):
    gabor_bank = np.zeros((hp.n_gabor, hp.gabor_size, hp.gabor_size, 1))
    for i in range(hp.n_gabor):
        gabor_bank[i, :, :, 0] = gabor_kernel(frequency=hp.gabor_freq[i], theta=hp.gabor_theta[i], bandwidth=hp.gabor_bandwidth[i])
    gabor_bank = gabor_bank.astype(np.float32)
    gabor_bank = tf.convert_to_tensor(gabor_bank)
    gabor_bank = tf.transpose(gabor_bank, perm=[1, 2, 3, 0])
    gabor_bank = tf.keras.layers.Conv2D(filters=hp.n_gabor, kernel_size=(hp.gabor_size, hp.gabor_size), padding='same', kernel_initializer=tf.keras.initializers.Constant(value=gabor_bank), trainable=False)
    return gabor_bank


#second functioin convolves a batch of images with a bank of filters
def convolve_gabor_bank(images, gabor_bank):

#function receives a batch of coordinates and a batch of update vectors and returns a batch of updated coordinates
#if updated coordinates are out of bounds, they are set to the closest valid value
def update_coordinates(coordinates, updates, min_coordinates, max_coordinates):
    updated_coordinates = coordinates + updates
    updated_coordinates = np.maximum(updated_coordinates, min_coordinates)
    updated_coordinates = np.minimum(updated_coordinates, max_coordinates)
    return updated_coordinates

#class for generic RL environment
class Environment:
    def __init__(self, hp):
        self.hp = hp
        self.reset()

    def reset(self):
        pass

    def step(self, action):
        pass

    def get_state(self):
        pass

    def get_reward(self):
        pass