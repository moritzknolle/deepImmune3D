import numpy as np


# Preprocesses multi channel images. Transforms from range [1-256] to [0-1] and optionally removes channel 5 (CD31+ channel)
def preprocess(data_arr:np.ndarray, remove_channel_five:bool=True, t:int=150):
    """ Perform preprocessing for deep learning model. Transforms image data from range [1-256] to [0-1] and optionally removes channel 5 (CD31+ channel)
        Args:
            data_arr: input image data to preprocess
            remove_channel_five: Boolean value to determine whether CD31+ chanel should be removed
            t: threshold for clipping
        Returns:
            preprocessed np.ndarray
    """
    if remove_channel_five and data_arr.shape[-1]==6:
        data_arr = np.delete(data_arr, 4, axis=-1)
    out = np.clip(data_arr, a_min=0, a_max=t)
    out = np.divide(out, t)
    return out