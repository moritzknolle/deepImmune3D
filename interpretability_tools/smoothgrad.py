import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tf_explain.utils.display import grid_display
from tf_explain.utils.image import transform_to_normalized_grayscale
from tf_explain.utils.saver import save_grayscale

import matplotlib.pyplot as plt


def min_max_scale(tensor):
    out_tensor = tf.math.divide(
        tf.math.subtract(tensor, tf.reduce_min(tensor)),
        tf.math.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor)),
    )
    return out_tensor


class SmoothGrad:

    """
    Compute SmoothGrad interpretability maps for a given input image
    Paper: [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)
    adapted from https://github.com/sicara/tf-explain
    """

    def explain(
        self,
        data,
        model,
        class_index,
        num_samples=5,
        noise=1.0,
        ch_specific_grads=False,
        transform_range=False,
    ):
        """ Computes and returns SmoothGrad gradient attribution map for given input data
            Args:
                data: data to analyse in form of numpy array or tf-Tensor
                model: tf.keras.models.Model instance to perform analysis with
                class_index: class index of data sample
                num_samples: the number of noisy samples to compute and average the gradient attribution maps with
                noise: noise standard deviation to apply
                ch_specific_grads: whether to compute channel specific gradient attribution maps
                transform_range: whether to scale output to [0-255] range
        """
        noisy_images = SmoothGrad.generate_noisy_images(data, num_samples, noise)
        smoothed_gradients = SmoothGrad.get_averaged_gradients(
            noisy_images, model, class_index, num_samples
        )

        if ch_specific_grads:
            # normalize
            print("returning channel specific interpretability map")
            normalized_grads = min_max_scale(tf.abs(smoothed_gradients)).numpy()
            out_grad = normalized_grads
            out_gradximg = (normalized_grads * data).squeeze()
            out_gradximg = min_max_scale(out_gradximg).numpy()

        else:
            # sum reduction along channel axis -> normalize
            print("returning channel unspecific interpretability map")
            reduced_grads = tf.reduce_sum(tf.abs(smoothed_gradients), axis=-1)
            normalized_grads = min_max_scale(reduced_grads).numpy()
            out_grad = normalized_grads
            out_gradximg = (
                (normalized_grads * tf.reduce_mean(data, axis=-1)).numpy().squeeze()
            )
            out_grad = np.repeat(
                np.expand_dims(out_grad, axis=-1), repeats=data.shape[-1], axis=-1,
            )
            out_gradximg = np.repeat(
                np.expand_dims(out_gradximg, axis=-1), repeats=data.shape[-1], axis=-1,
            )
            out_gradximg = min_max_scale(out_gradximg).numpy()
        if transform_range:
            out_grad = (out_grad * 255).astype(np.uint8)
            out_gradximg = (out_gradximg * 255).astype(np.uint8)
        return out_grad, out_gradximg

    @staticmethod
    def generate_noisy_images(images, num_samples, noise):
        repeated_images = np.repeat(images, num_samples, axis=0)
        noise = np.random.normal(0, noise, repeated_images.shape).astype(np.float32)

        return repeated_images + noise

    @staticmethod
    @tf.function
    def get_gradients(inputs, targets, model):
        with tf.GradientTape() as tape:
            inputs = tf.cast(inputs, tf.float32)
            tape.watch(inputs)
            predictions = model(inputs)
            loss = tf.keras.losses.binary_crossentropy(targets, predictions)
        grads = tape.gradient(loss, inputs)
        return grads

    @staticmethod
    def get_averaged_gradients(noisy_images, model, class_index, num_samples):
        expected_output = np.expand_dims(
            np.array([class_index], dtype=np.float32), axis=0
        )
        grad_list = []
        for sample in tqdm(noisy_images):
            sample_grad = SmoothGrad.get_gradients(
                inputs=sample[None], targets=expected_output, model=model
            )
            grad_list.append(sample_grad.numpy())
        grads = np.stack(grad_list, axis=0).squeeze()
        averaged_grads = np.mean(grads, axis=0)

        return averaged_grads
