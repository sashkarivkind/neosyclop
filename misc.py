#python file that contains miscellaneous functions
import numpy as np
import copy
#keras model that adds global average pooling
#to the end of a model and then two fully connected layers followed by softmax
#input_shape is the shape of the input image
#num_classes is the number of classes
#output_scale is the scale of the output image
def add_global_average_pooling(model, num_classes, output_scale=1.0):
    input = model.input
    output = model.output
    output = GlobalAveragePooling2D()(output)
    output = Flatten()(output)
    #dropout
    output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(64, activation='relu')(output)
    output = Dense(num_classes, activation='softmax')(output)
    output = tf.multiply(output, output_scale)
    model = Model(input, output)
    return model


#function that receives a list or array and
#returns a smoothed version of the list or array
#the smoothing is done by taking the average of
#the current element and the previous and next elements
#x is the list or array
#n is the window size
#axis is the axis along which the smoothing is done only for arrays
def smooth(x, n, axis=None):
    if axis is None:
        return np.convolve(x, np.ones((n,))/n, mode='valid')
    else:
        return np.apply_along_axis(lambda m: np.convolve(m, np.ones((n,))/n, mode='valid'), axis, x)

#keras model that receives a multispectral image applies a convolution layer
#with optional activation function and returns an RGB image
#input_shape is the shape of the input image
#kernel_size is the size of the convolution kernel
#activation is the activation function
#output_scale is the scale of the output image
def multispectral_convolution(input_shape, kernel_size, activation=None, output_scale=1.0):
    input = Input(shape=input_shape)
    conv = Conv2D(3, kernel_size, activation=activation, padding='same')(input)
    output = tf.multiply(conv, output_scale)
    return Model(input, output)

#function that prints progress bar
#iteration is the current iteration
#total is the total number of iterations
#prefix is the prefix of the progress bar
#suffix is the suffix of the progress bar
#decimals is the number of decimals to be printed
#length is the length of the progress bar
#fill is the character to be used to fill the progress bar
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ('{0:.' + str(decimals) + 'f}').format(100*(iteration/float(total)))
    filled_length = int(length*iteration//total)
    bar = fill*filled_length + '-'*(length-filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()

#function that receives an object whose attributes are default parameters
#and an object whose attributes are defined parameters and returns a dictionary
#of parameters with the default parameters updated with the user defined parameters
#default_params is an object whose attributes are default parameters
#user_params is an object whose attributes are user defined parameters
def update_with_defaults(default_params=None, user_params=None):
    params = copy.copy(default_params)
    for key, value in user_params.__dict__.items():
        #if the key is not in the default parameters
        if key not in params.__dict__:
            #raise warning
            print('Warning: parameter {} is not in the default parameters'.format(key))
        params.__dict__[key] = value
    return params






    # params = default_params.copy()
    # for key, value in user_params.items():
    #     #if the key is not in the default parameters
    #     if key not in params: #raise warning
    #         print('Warning: parameter {} is not in the default parameters'.format(key))
    #     params[key] = value
    # return params