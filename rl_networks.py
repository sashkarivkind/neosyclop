import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Concatenate
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers

#create actor model in Keras
# model receives the following inputs:
# image, defined by image_h and image_h
# spectral_density_size, defined by spectral_density_size
# location_history, defined by location_history_size
# timestep, defined by timestep_size
# image is fed through a convolutional network, other inputs are fed through a dense network
# the outputs of the two networks are concatenated and fed through a dense network
# the output of this network is the action is of dimension action_size


def create_actor_model(image_h, image_w, spectral_density_size, location_history_size, timestep_size, action_size, output_scale=3.0,
                        image_out_size=32,
                        spectral_density_out_size=32,
                        location_history_out_size=32,
                        timestep_out_size=32):
    # image input
    image_input = Input(shape=(image_h, image_w, 1), name='image_input')
    image_conv = Conv2D(32, (7, 7), activation='relu', padding='same')(image_input)
    image_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(image_input)
    image_conv = MaxPooling2D((4, 4), padding='same')(image_conv) # 56x56
    #layer normalization along the height and width dimensions
    image_conv = tf.keras.layers.LayerNormalization(axis=[1, 2])(image_conv)

    image_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(image_conv)
    image_conv = MaxPooling2D((2, 2), padding='same')(image_conv) # 28x28
    #layer normalization along the height and width dimensions
    image_conv = tf.keras.layers.LayerNormalization(axis=[1, 2])(image_conv)

    image_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(image_conv)
    image_conv = MaxPooling2D((2, 2), padding='same')(image_conv) # 14x14
    image_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(image_conv)
    image_conv = MaxPooling2D((2, 2), padding='same')(image_conv) # 7x7
    image_conv = Conv2D(64, (1, 1), activation='relu', padding='same')(image_conv)
    image_conv = Flatten()(image_conv) # 7x7x64=3136
    image_out = Dense(image_out_size, activation='relu')(image_conv)

    #normalise image_out
    image_out = tf.math.l2_normalize(image_out, axis=1)
    image_out_size = 3136
    # spectral density input
    spectral_density_input = Input(shape=(spectral_density_size,), name='spectral_density_input')
    spectral_density_out = Dense(spectral_density_out_size, activation='relu')(spectral_density_input)
    # normalise spectral_density_out
    spectral_density_out = tf.math.l2_normalize(spectral_density_out, axis=1)

    # location history input
    location_history_input = Input(shape=(2, location_history_size // 2), name='location_history_input')
    location_history_input_ = Flatten()(location_history_input)
    location_history_out = Dense(location_history_out_size, activation='relu')(location_history_input_)
    # normalise location_history_out
    location_history_out = tf.math.l2_normalize(location_history_out, axis=1)

    # timestep input
    timestep_input = Input(shape=(timestep_size,), name='timestep_input')
    timestep_out = Dense(timestep_out_size, activation='relu')(timestep_input)
    # normalise timestep_out
    timestep_out = tf.math.l2_normalize(timestep_out, axis=1)

    # concatenate all inputs
    #prior to concatenation, all inputs are fed normalised according to their sqrt of output size
    image_out = tf.multiply(image_out, 1.0 / np.sqrt(image_out_size))
    spectral_density_out = tf.multiply(spectral_density_out, 1.0 / np.sqrt(spectral_density_out_size))
    location_history_out = tf.multiply(location_history_out, 1.0 / np.sqrt(location_history_out_size))
    timestep_out = tf.multiply(timestep_out, 1.0 / np.sqrt(timestep_out_size))

    # concatenate all inputs
    concatenated = Concatenate()([image_out, spectral_density_out, location_history_out, timestep_out])
    # dense layer
    concatenated = Dense(512, activation='relu')(concatenated)
    concatenated = Dense(512, activation='relu')(concatenated)
    concatenated_dense = Dense(32, activation='relu')(concatenated)
    # output layer
    output = Dense(action_size, activation='tanh')(concatenated_dense)
    output = tf.multiply(output, output_scale, name='output')
    # create model
    model = Model(inputs=[image_input, spectral_density_input, location_history_input, timestep_input], outputs=output)
    return model


# create critic model in Keras
# model receives the following inputs:
# image, defined by image_h and image_h
# spectral_density_size, defined by spectral_density_size
# location_history, defined by location_history_size
# timestep, defined by timestep_size
# action, defined by action_size
# image is fed through a convolutional network, other inputs are fed through a dense network
# the outputs of the two networks are concatenated and fed through a dense network
# the output of this network is the Q-value of dimension 1

def create_critic_model(image_h, image_w, spectral_density_size, location_history_size, timestep_size, action_size,
                        image_out_size=32,
                        spectral_density_out_size=32,
                        location_history_out_size=32,
                        timestep_out_size=32,
                        action_out_size=32):
    # image input
    image_input = Input(shape=(image_h, image_w, 1), name='image_input')
    image_conv = Conv2D(32, (7, 7), activation='relu', padding='same')(image_input)
    image_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(image_input)
    image_conv = MaxPooling2D((4, 4), padding='same')(image_conv) # 56x56
    #layer normalization along the height and width dimensions
    image_conv = tf.keras.layers.LayerNormalization(axis=[1, 2])(image_conv)

    image_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(image_conv)
    image_conv = MaxPooling2D((2, 2), padding='same')(image_conv) # 28x28
    #layer normalization along the height and width dimensions
    image_conv = tf.keras.layers.LayerNormalization(axis=[1, 2])(image_conv)

    image_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(image_conv)
    image_conv = MaxPooling2D((2, 2), padding='same')(image_conv) # 14x14
    image_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(image_conv)
    image_conv = MaxPooling2D((2, 2), padding='same')(image_conv) # 7x7
    image_conv = Conv2D(64, (1, 1), activation='relu', padding='same')(image_conv)
    image_conv = Flatten()(image_conv) # 7x7x64=3136
    image_out = Dense(image_out_size, activation='relu')(image_conv)

    #normalise image_out
    image_out = tf.math.l2_normalize(image_out, axis=1)
    image_out_size = 3136
    # spectral density input
    spectral_density_input = Input(shape=(spectral_density_size,), name='spectral_density_input')
    spectral_density_out = Dense(spectral_density_out_size, activation='relu')(spectral_density_input)
    # normalise spectral_density_out
    spectral_density_out = tf.math.l2_normalize(spectral_density_out, axis=1)

    # location history input
    location_history_input = Input(shape=(2, location_history_size // 2), name='location_history_input')
    location_history_input_ = Flatten()(location_history_input)
    location_history_out = Dense(location_history_out_size, activation='relu')(location_history_input_)
    # normalise location_history_out
    location_history_out = tf.math.l2_normalize(location_history_out, axis=1)

    # timestep input
    timestep_input = Input(shape=(timestep_size,), name='timestep_input')
    timestep_out = Dense(timestep_out_size, activation='relu')(timestep_input)
    # normalise timestep_out
    timestep_out = tf.math.l2_normalize(timestep_out, axis=1)

    # action input
    action_input = Input(shape=(action_size,), name='action_input')
    action_out = Dense(action_out_size, activation='relu')(action_input)
    # normalise action_out
    action_out = tf.math.l2_normalize(action_out, axis=1)

    # concatenate all inputs
    #prior to concatenation, all inputs are fed normalised according to their sqrt of output size
    image_out = tf.multiply(image_out, 1.0 / np.sqrt(image_out_size))
    spectral_density_out = tf.multiply(spectral_density_out, 1.0 / np.sqrt(spectral_density_out_size))
    location_history_out = tf.multiply(location_history_out, 1.0 / np.sqrt(location_history_out_size))
    timestep_out = tf.multiply(timestep_out, 1.0 / np.sqrt(timestep_out_size))
    action_out = tf.multiply(action_out, 1.0 / np.sqrt(action_out_size))

    concatenated = Concatenate()([image_out, spectral_density_out, location_history_out, timestep_out, action_out])
    # dense layer
    concatenated_dense = Dense(32, activation='relu')(concatenated)
    # output layer
    output = Dense(1, activation='linear')(concatenated_dense)
    # create model
    model = Model(inputs=[image_input, spectral_density_input, location_history_input, timestep_input, action_input], outputs=output)
    return model


def policy(tf_prev_state, actor_model, lower_bound, upper_bound):
    # Pass the previous state through the actor model
    raw_action = actor_model(tf_prev_state)

    # Clip the action to the specified bounds
    clipped_action = tf.clip_by_value(raw_action, lower_bound, upper_bound)

    return clipped_action


# a test network for the actor, suitable for the mountain car environment
def create_actor_network_test_v1(input_shape, action_range):
    model = models.Sequential()

    model.add(layers.Dense(24, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(1, activation='tanh'))

    # Scale the output to the action range
    model.add(layers.Lambda(lambda x: x * action_range))

    return model