import tensorflow as tf
import numpy as np


@tf.function
def flip_img(x, p=0.5):
    """ Flips input tensor across random axes
        Args:
            x: input tensor
            p: probability of augmentation being applied
    """
    dims = [i for i in range(len(list(x.shape))-1)]
    if tf.random.uniform([]) < p:
        ax = int(np.random.choice(dims))
        x = tf.reverse(x, axis=[ax])
    return x

@tf.function
def random_gauss_img(x, std, p=0.5):
    """ Adds random Gaussian noise to input tensor
        Args:
            x: input tensor
            std: standard deviation of the Gaussian noise
            p: probability of augmentation being applied
    """
    if tf.random.uniform([]) < p:
        noise = tf.random.normal(x.shape, stddev=std)
        x += noise
    return x
    
def Random_Flip(prob=0.5):
    """Random Flipping Lamda Layer Wrapper"""
    return tf.keras.layers.Lambda(lambda b: flip_img(b, prob))

def Random_Gaussian_Noise(prob=0.5, std=0.2):
    """Random Gaussian Noise Lamda Layer Wrapper"""
    return tf.keras.layers.Lambda(lambda b: random_gauss_img(b, std, prob))

augmentation = tf.keras.Sequential([
    Random_Flip(),
    Random_Gaussian_Noise(std=0.1)
])

def prepare_dataset(ds, augment=False, shuffle=True, batch_size=1):
    """ Maps the above defined sequential augmentation pipeline to a given dataset
        Args:
            ds: input dataset
            augment: Boolean value to determine whether to apply augmentation
            shuffle: whether to shuffle dataset
            batch_size: batch size
    """
    if augment:
        ds = ds.map(lambda x, y: (augmentation(x), y), 
              num_parallel_calls=-1)
    if shuffle:
        ds = ds.shuffle(100)
    ds = ds.batch(batch_size)
    return ds.prefetch(1)

            
