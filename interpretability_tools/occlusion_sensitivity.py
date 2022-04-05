import numpy as np
import cv2
import math
from tf_explain.utils.display import grid_display, heatmap_display
import matplotlib.pyplot as plt
import tensorflow as tf


class OcclusionSensitivity:
    """
    Compute occlusion sensitivity interpretability maps for a given input image
    adapted from https://github.com/sicara/tf-explain
    """

    def __init__(self, batch_size=None):
        self.batch_size = batch_size

    def apply_grey_patch(self, image, top_left_x, top_left_y, patch_size):
        patched_image = np.array(image, copy=True)
        patched_image[
            top_left_y : top_left_y + patch_size, top_left_x : top_left_x + patch_size :
        ] = 127.5

        return patched_image

    def explain(
        self, images, model, class_index, patch_size,
    ):
        """ Computes and returns occlusion sensitivty interpretability map for given input images
            Args:
                images: data to analyse in form of numpy array or tf-Tensor
                model: tf.keras.models.Model instance to perform analysis with
                class_index: class index of data sample
                patch_size: size of patch to perform occlusion with
        """

        sensitivity_maps = np.array(
            [
                self.get_sensitivity_map(model, image, class_index, patch_size)
                for image in images
            ]
        )
        return sensitivity_maps[0]

    def get_sensitivity_map(self, model, image, class_index, patch_size):
        sensitivity_map = np.zeros(
            (
                math.ceil(image.shape[0] / patch_size),
                math.ceil(image.shape[1] / patch_size),
            )
        )

        patches = [
            self.apply_grey_patch(image, top_left_x, top_left_y, patch_size)
            for index_x, top_left_x in enumerate(range(0, image.shape[0], patch_size))
            for index_y, top_left_y in enumerate(range(0, image.shape[1], patch_size))
        ]

        coordinates = [
            (index_y, index_x)
            for index_x in range(sensitivity_map.shape[1])
            for index_y in range(sensitivity_map.shape[0])
        ]

        predictions = model.predict(np.array(patches), batch_size=self.batch_size)
        target_class_predictions = [
            prediction[class_index] for prediction in predictions
        ]

        for (index_y, index_x), confidence in zip(
            coordinates, target_class_predictions
        ):
            sensitivity_map[index_y, index_x] = 1 - confidence

        return cv2.resize(sensitivity_map, image.shape[0:2])

