import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Concatenate
from tensorflow.keras.models import Model


#create actor model in Keras
# model receives the following inputs:
# image, defined by image_h and image_h
# spectral_density_size, defined by spectral_density_size
# location_history, defined by location_history_size
# timestep, defined by timestep_size
# image is fed through a convolutional network, other inputs are fed through a dense network
# the outputs of the two networks are concatenated and fed through a dense network
# the output of this network is the action is of dimension action_size


def create_actor_model(image_h, image_w, spectral_density_size, location_history_size, timestep_size, action_size, output_scale=3.0):
    # image input
    image_input = Input(shape=(image_h, image_w, 1), name='image_input')
    image_conv = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    image_conv = MaxPooling2D((2, 2), padding='same')(image_conv)
    image_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(image_conv)
    image_conv = MaxPooling2D((2, 2), padding='same')(image_conv)
    image_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(image_conv)
    image_conv = Flatten()(image_conv)
    # spectral density input
    spectral_density_input = Input(shape=(spectral_density_size,), name='spectral_density_input')
    spectral_density_dense = Dense(32, activation='relu')(spectral_density_input)
    # location history input
    location_history_input = Input(shape=(2,location_history_size//2), name='location_history_input')
    location_history_input_ = Flatten()(location_history_input)
    location_history_dense = Dense(32, activation='relu')(location_history_input_)
    # timestep input
    timestep_input = Input(shape=(timestep_size,), name='timestep_input')
    timestep_dense = Dense(32, activation='relu')(timestep_input)
    # concatenate all inputs
    concatenated = Concatenate()([image_conv, spectral_density_dense, location_history_dense, timestep_dense])
    # dense layer
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

def create_critic_model(image_h, image_w, spectral_density_size, location_history_size, timestep_size, action_size):
    # image input
    image_input = Input(shape=(image_h, image_w, 1), name='image_input')
    image_conv = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    image_conv = MaxPooling2D((2, 2), padding='same')(image_conv)
    image_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(image_conv)
    image_conv = MaxPooling2D((2, 2), padding='same')(image_conv)
    image_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(image_conv)
    image_conv = Flatten()(image_conv)
    # spectral density input
    spectral_density_input = Input(shape=(spectral_density_size,), name='spectral_density_input')
    spectral_density_dense = Dense(32, activation='relu')(spectral_density_input)
    # location history input
    location_history_input = Input(shape=(2,location_history_size // 2), name='location_history_input')
    location_history_input_ = Flatten()(location_history_input)
    location_history_dense = Dense(32, activation='relu')(location_history_input_)
    # timestep input
    timestep_input = Input(shape=(timestep_size,), name='timestep_input')
    timestep_dense = Dense(32, activation='relu')(timestep_input)
    # action input
    action_input = Input(shape=(action_size,), name='action_input')
    action_dense = Dense(32, activation='relu')(action_input)
    # concatenate all inputs
    concatenated = Concatenate()([image_conv, spectral_density_dense, location_history_dense, timestep_dense, action_dense])
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