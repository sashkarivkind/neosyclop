#python file that contains miscellaneous functions
import numpy as np

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

