import numpy as np
import tensorflow as tf
from tqdm import tqdm
import gc

def monte_carlo_predict(model, data, num_samples, batch_size=1):
    """ Performs a Monte-Carlo prediction given a non-deterministic model
        Args:
            model: tf.keras.models.Model instance to perform predictions with
            data: data to perform prediction with
            batch_size: batch size to perform forward passes with 
        """
    preds = []
    for _ in tqdm(range(num_samples)):
        preds.append(model.predict(data, batch_size=batch_size))
    gc.collect()
    mc_pred = np.stack(preds, axis=0)
    return mc_pred

class GradientAccumulator(tf.keras.Model):
    """ Gradient Accumulation wrapper class to enable training with larger batch sizes.
        adapted from https://stackoverflow.com/questions/66472201/gradient-accumulation-with-custom-model-fit-in-tf-keras"""

    def __init__(self, effective_batch_size:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gradients = tf.constant(effective_batch_size, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.total_grads = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

    def train_step(self, data):
        """" Performs a custom train step, aggregates gradients and performs gradient descent update step every n_accum_step
            """
        self.n_acum_step.assign_add(1)

        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        for i in range(len(self.total_grads)):
            self.total_grads[i].assign_add(gradients[i])
 
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        for i in range(len(self.total_grads)):
            grad = self.total_grads[i]
            self.total_grads[i].assign(grad/tf.cast(self.n_gradients, tf.float32))
        self.optimizer.apply_gradients(zip(self.total_grads, self.trainable_variables))
        self.n_acum_step.assign(0)
        for i in range(len(self.total_grads)):
            self.total_grads[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))
