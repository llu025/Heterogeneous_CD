import os
import gc

# Set loglevel to suppress tensorflow GPU messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import datasets
from change_detector import ChangeDetector
from image_translation import CouplingNetwork
from config import get_config_SCCN
from decorators import image_to_tensorboard, timed
from tqdm import trange
import numpy as np


class SCCN(ChangeDetector):
    def __init__(self, translation_spec, **kwargs):
        """
                Input:
                    translation_spec - dict with keys 'enc_X', 'enc_Y', 'dec_X', 'dec_Y'.
                                       Values are passed as kwargs to the
                                       respective ImageTranslationNetwork's
                    cycle_lambda=2 - float, loss weight
                    cross_lambda=1 - float, loss weight
                    l2_lambda=1e-3 - float, loss weight
                    kernels_lambda - float, loss weight
                    learning_rate=1e-5 - float, initial learning rate for
                                         ExponentialDecay
                    clipnorm=None - gradient norm clip value, passed to
                                    tf.clip_by_global_norm if not None
                    logdir=None - path to log directory. If provided, tensorboard
                                  logging of training and evaluation is set up at
                                  'logdir/timestamp/' + 'train' and 'evaluation'
        """

        super().__init__(**kwargs)

        self.l2_lambda = kwargs.get("l2_lambda", 1e-6)
        self.Lambda = kwargs.get("kernels_lambda", 1)

        # encoders of X and Y
        self._enc_x = CouplingNetwork(
            **translation_spec["enc_X"], name="enc_X", l2_lambda=self.l2_lambda,
        )
        self._enc_y = CouplingNetwork(
            **translation_spec["enc_Y"], name="enc_Y", l2_lambda=self.l2_lambda,
        )

        # decoder of X and Y
        self._dec_x = CouplingNetwork(
            **translation_spec["dec_X"],
            name="dec_X",
            decoder=True,
            l2_lambda=self.l2_lambda,
        )
        self._dec_y = CouplingNetwork(
            **translation_spec["dec_Y"],
            name="dec_Y",
            decoder=True,
            l2_lambda=self.l2_lambda,
        )

        self.loss_object = tf.keras.losses.MeanSquaredError()

        self.train_metrics["code"] = tf.keras.metrics.Sum(name="krnls MSE sum")
        self.train_metrics["l2"] = tf.keras.metrics.Sum(name="l2 MSE sum")

    @image_to_tensorboard()
    def enc_x(self, inputs, training=False):
        """ Wraps encoder call for TensorBoard printing and image save """
        return self._enc_x(inputs, training)

    @image_to_tensorboard()
    def enc_y(self, inputs, training=False):
        return self._enc_y(inputs, training)

    @image_to_tensorboard()
    def dec_x(self, inputs, training=False):
        return self._dec_x(inputs, training)

    @image_to_tensorboard()
    def dec_y(self, inputs, training=False):
        return self._dec_y(inputs, training)

    def __call__(self, inputs, training=False, pretraining=False):
        x, y = inputs
        tf.debugging.Assert(tf.rank(x) == 4, [x.shape])
        tf.debugging.Assert(tf.rank(y) == 4, [y.shape])

        if training:
            x_code, y_code = self._enc_x(x, training), self._enc_y(y, training)
            if pretraining:
                x_tilde, y_tilde = (
                    self._dec_x(x_code, training),
                    self._dec_y(y_code, training),
                )

                retval = [x_tilde, y_tilde]
            else:
                retval = [x_code, y_code]
            return retval

        else:
            x_code, y_code = self.enc_x(x, name="x_code"), self.enc_y(y, name="y_code")
            difference_img = self._domain_difference_img(x_code, y_code)
            retval = difference_img

        return retval

    def _train_step(self, x, y, clw):
        """
        Input:
        x - tensor of shape (bs, ps_h, ps_w, c_x)
        y - tensor of shape (bs, ps_h, ps_w, c_y)
        clw - cross_loss_weight, tensor of shape (bs, ps_h, ps_w, 1)
        """
        with tf.GradientTape() as tape:
            x_code, y_code = self([x, y], training=True)

            l2_loss = sum(self._enc_x.losses) + sum(self._enc_y.losses)
            targets = self._enc_x.trainable_variables + self._enc_y.trainable_variables
            code_loss = self.loss_object(x_code, y_code, clw)
            tot_loss = code_loss - self.Lambda * tf.reduce_mean(clw)
            gradients = tape.gradient(tot_loss + l2_loss, targets)
            if self.clipnorm is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, self.clipnorm)

            self._optimizer_all.apply_gradients(zip(gradients, targets))

        self.train_metrics["code"].update_state(code_loss)
        self.train_metrics["l2"].update_state(l2_loss)

    @timed
    def pretrain(
        self, training_dataset, preepochs, batches, batch_size, **kwargs,
    ):
        """
        Input:
        x - tensor of shape (bs, ps_h, ps_w, c_x)
        y - tensor of shape (bs, ps_h, ps_w, c_y)
        clw - cross_loss_weight, tensor of shape (bs, ps_h, ps_w, 1)
        """
        tf.print("Pretrain")

        for epoch in trange(preepochs):
            for i, batch in zip(range(batches), training_dataset.batch(batch_size)):
                x, y, _ = batch
                with tf.GradientTape() as tape:
                    x_tilde, y_tilde = self([x, y], training=True, pretraining=True)
                    recon_x_loss = self.loss_object(x, x_tilde)
                    recon_y_loss = self.loss_object(y, y_tilde)
                    l2_loss = (
                        sum(self._enc_x.losses)
                        + sum(self._enc_y.losses)
                        + sum(self._dec_x.losses)
                        + sum(self._dec_y.losses)
                    )
                    total_loss = recon_x_loss + recon_y_loss + l2_loss
                    targets_pre = (
                        self._enc_x.trainable_variables
                        + self._enc_y.trainable_variables
                        + self._dec_x.trainable_variables
                        + self._dec_y.trainable_variables
                    )

                    gradients_pre = tape.gradient(total_loss, targets_pre)

                    if self.clipnorm is not None:
                        gradients_pre, _ = tf.clip_by_global_norm(
                            gradients_pre, self.clipnorm
                        )
                    self._optimizer_k.apply_gradients(zip(gradients_pre, targets_pre))
        tf.print("Pretrain done")


def test(DATASET="Texas", CONFIG=None):
    """
    1. Fetch data (x, y, change_map)
    2. Compute/estimate A_x and A_y (for patches)
    3. Compute change_prior
    4. Define dataset with (x, A_x, y, A_y, p). Choose patch size compatible
       with affinity computations.
    5. Train CrossCyclicImageTransformer unsupervised
        a. Evaluate the image transformations in some way?
    6. Evaluate the change detection scheme
        a. change_map = threshold [(x - f_y(y))/2 + (y - f_x(x))/2]
    """
    if CONFIG is None:
        CONFIG = get_config_SCCN(DATASET)
    x_im, y_im, EVALUATE, (C_X, C_Y) = datasets.fetch(DATASET, **CONFIG)
    if tf.test.is_gpu_available() and not CONFIG["debug"]:
        C_CODE = CONFIG["C_CODE"]
        TRANSLATION_SPEC = {
            "enc_X": {
                "input_chs": C_X,
                "filter_spec": [C_CODE, C_CODE, C_CODE, C_CODE],
            },
            "enc_Y": {
                "input_chs": C_Y,
                "filter_spec": [C_CODE, C_CODE, C_CODE, C_CODE],
            },
            "dec_X": {"input_chs": C_CODE, "filter_spec": [C_X]},
            "dec_Y": {"input_chs": C_CODE, "filter_spec": [C_Y]},
        }
    else:
        C_CODE = 1
        TRANSLATION_SPEC = {
            "enc_X": {"input_chs": C_X, "filter_spec": [C_CODE]},
            "enc_Y": {"input_chs": C_Y, "filter_spec": [C_CODE]},
            "dec_X": {"input_chs": C_CODE, "filter_spec": [C_X]},
            "dec_Y": {"input_chs": C_CODE, "filter_spec": [C_Y]},
        }
    cd = SCCN(TRANSLATION_SPEC, **CONFIG)
    training_time = 0
    Pu = tf.expand_dims(tf.ones(x_im.shape[:-1], dtype=tf.float32), -1)
    TRAIN = tf.data.Dataset.from_tensor_slices((x_im, y_im, Pu))
    TRAIN = TRAIN.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    training_time, _ = cd.pretrain(EVALUATE, evaluation_dataset=EVALUATE, **CONFIG)
    epochs = CONFIG["epochs"]
    CONFIG.update(epochs=1)
    for epoch in trange(epochs):
        tr_time, _ = cd.train(TRAIN, evaluation_dataset=EVALUATE, **CONFIG)
        training_time += tr_time
        if epoch > 10:
            for x, y, _ in EVALUATE.batch(1):
                Pu = 1.0 - tf.cast(cd._change_map(cd([x, y])), dtype=tf.float32)
            del TRAIN
            gc.collect()
            TRAIN = tf.data.Dataset.from_tensor_slices((x_im, y_im, Pu))
            TRAIN = TRAIN.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    cd.final_evaluate(EVALUATE, **CONFIG)
    final_kappa = cd.metrics_history["cohens kappa"][-1]
    timestamp = cd.timestamp
    epoch = cd.epoch.numpy()
    return final_kappa, epoch, training_time, timestamp


if __name__ == "__main__":
    test("Texas")
    test("California")
