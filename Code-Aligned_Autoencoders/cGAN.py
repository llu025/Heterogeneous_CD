import os

# Set loglevel to suppress tensorflow GPU messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

run_opts = tf.compat.v1.RunOptions(
    report_tensor_allocations_upon_oom=True, trace_level=3
)

from tensorflow.python.keras.utils import losses_utils
import datasets
from change_detector import ChangeDetector
from image_translation import Generator, Discriminator
from config import get_config_CGAN
from decorators import image_to_tensorboard
import numpy as np
from tqdm import trange
from change_priors import image_in_patches


class LogError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        losses = tf.math.log(tf.abs(y_true - y_pred))
        return losses


class CGAN(ChangeDetector):
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
        self.ps = kwargs.get("patch_size", 5)
        self.Lambda = kwargs.get("Lambda", 5)

        # Generator and Approximator of SAR data
        self._gen = Generator(
            **translation_spec["Generator"], name="Gen", l2_lambda=self.l2_lambda,
        )
        self._approx = Generator(
            **translation_spec["Approximator"], name="Approx", l2_lambda=self.l2_lambda,
        )

        # Discriminator
        self._discr = Discriminator(
            **translation_spec["Discriminator"], name="Discr", l2_lambda=self.l2_lambda
        )

        self.L2_loss = tf.keras.losses.MeanSquaredError()
        self.L1_loss = tf.keras.losses.MeanAbsoluteError()
        self.log_loss = LogError()

        self.train_metrics["Generation"] = tf.keras.metrics.Sum(name="Generate")
        self.train_metrics["Approximation"] = tf.keras.metrics.Sum(name="Approximate")
        self.train_metrics["Discrimination"] = tf.keras.metrics.Sum(name="Discriminate")
        self.train_metrics["Fooling"] = tf.keras.metrics.Sum(name="Fool")
        self.train_metrics["total_gen"] = tf.keras.metrics.Sum(name="total_gen sum")
        self.train_metrics["total_dis"] = tf.keras.metrics.Sum(name="total_dis sum")
        self.train_metrics["l2"] = tf.keras.metrics.Sum(name="l2 MSE sum")

        # Track kernel loss history for use in early stopping
        self.metrics_history["krnls"] = []

    @image_to_tensorboard()
    def gen(self, input, gen_chs):
        """ Wraps encoder call for TensorBoard printing and image save """
        h, w, ch = input.shape[1], input.shape[2], input.shape[-1]
        input = image_in_patches(input, self.ps)
        nb = input.shape[0] // 200 + (input.shape[0] % 200)
        tmp = tf.zeros([1, self.ps, self.ps, gen_chs], dtype=tf.float32)
        for i in range(nb):
            start = i * 200
            stop = tf.reduce_min([input.shape[0], start + 200])
            tmp = tf.concat([tmp, self._gen(input[start:stop])], 0)
        tmp = tf.nn.depth_to_space(
            tf.reshape(tmp[1:], [1, h // self.ps, w // self.ps, -1]), self.ps
        )
        return tf.reshape(tmp, [1, h, w, -1])

    @image_to_tensorboard()
    def approx(self, input):
        """ Wraps encoder call for TensorBoard printing and image save """
        h, w, ch = input.shape[1], input.shape[2], input.shape[-1]
        input = image_in_patches(input, self.ps)
        nb = input.shape[0] // 200 + (input.shape[0] % 200)
        tmp = tf.zeros([1, self.ps, self.ps, ch], dtype=tf.float32)
        for i in range(nb):
            start = i * 200
            stop = tf.reduce_min([input.shape[0], start + 200])
            tmp = tf.concat([tmp, self._approx(input[start:stop])], 0)
        tmp = tf.nn.depth_to_space(
            tf.reshape(tmp[1:], [1, h // self.ps, w // self.ps, -1]), self.ps
        )
        return tf.reshape(tmp, [1, h, w, -1])

    def __call__(self, inputs, training=False):
        x, y = inputs

        if training:
            tf.debugging.Assert(tf.rank(x) == 4, [x.shape])
            tf.debugging.Assert(tf.rank(y) == 4, [y.shape])
            x_gen, x_approx = self._gen(y, training), self._approx(x, training)
            z_discr = self._discr(tf.concat([x_gen, x], 0))
            retval = [x_gen, x_approx, z_discr]

        else:
            x_gen = self.gen(y, x.shape[-1], name="x_gen")
            x_approx = self.approx(x, name="x_approx")
            difference_img = self._domain_difference_img(x_gen, x_approx)
            retval = difference_img
        return retval

    @tf.function
    def _train_step(self, x, y, clw):
        """
        Input:
        x - tensor of shape (bs, ps_h, ps_w, c_x)
        y - tensor of shape (bs, ps_h, ps_w, c_y)
        clw - cross_loss_weight, tensor of shape (bs, ps_h, ps_w, 1)
        """
        clw_g = tf.tile(tf.expand_dims(clw, -1), [1, self.ps])
        clw_d = tf.concat([clw_g, clw_g], 0)
        with tf.GradientTape(persistent=True) as tape:
            x_gen, x_approx, z_discr = self([x, y], training=True)
            gen_discr = z_discr[: z_discr.shape[0] // 2]
            labels_g = tf.ones(gen_discr.shape, dtype=tf.float32)
            labels_d = tf.concat([1.0 - labels_g, labels_g], 0)
            L1G_loss = self.L1_loss(x, x_gen, clw_g)
            L1A_loss = self.L1_loss(x_approx, x_gen, clw_g)

            fooling_loss = self.L2_loss(gen_discr, labels_g, clw_g)

            discr_loss = self.L2_loss(z_discr, labels_d, clw_d)
            tot_gen_loss = sum(self._gen.losses) + self.Lambda * L1G_loss + fooling_loss
            tot_approx_loss = L1A_loss + sum(self._approx.losses)
            tot_dis_loss = discr_loss + sum(self._discr.losses)
            tot_loss = [
                tot_gen_loss,
                tot_approx_loss,
                tot_dis_loss,
            ]
        l2_loss = (
            sum(self._gen.losses) + sum(self._approx.losses) + sum(self._discr.losses)
        )
        targets = [
            self._gen.trainable_variables,
            self._approx.trainable_variables,
            self._discr.trainable_variables,
        ]

        grads = []
        for i, v in enumerate(targets):
            grads += tape.gradient(tot_loss[i], v)
        targets = [item for sublist in targets for item in sublist]
        if self.clipnorm is not None:
            clipped_grads, _ = tf.clip_by_global_norm(grads, self.clipnorm)
        self._optimizer_all.apply_gradients(zip(clipped_grads, targets))

        self.train_metrics["Generation"].update_state(L1G_loss)
        self.train_metrics["Approximation"].update_state(L1A_loss)
        self.train_metrics["Discrimination"].update_state(discr_loss)
        self.train_metrics["Fooling"].update_state(fooling_loss)
        self.train_metrics["total_gen"].update_state(tot_gen_loss)
        self.train_metrics["total_dis"].update_state(tot_dis_loss)
        self.train_metrics["l2"].update_state(l2_loss)


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
        CONFIG = get_config_CGAN(DATASET)

    bs = CONFIG["batch_size"]
    ps = CONFIG["patch_size"]
    print(f"Loading {DATASET} data")

    x, y, EVALUATE, (C_X, C_Y), tot_patches = datasets.fetch_CGAN(DATASET, **CONFIG)
    Pu = tf.ones(x.shape[0], dtype=tf.float32)
    batches = tot_patches // bs + (tot_patches % bs != 0)
    CONFIG.update({"tot_patches": tot_patches, "batches": batches})
    if tf.config.list_physical_devices("GPU") and not CONFIG["debug"]:
        TRANSLATION_SPEC = {
            "Generator": {
                "shapes": [ps, C_Y],
                "filter_spec": [25, 100, 500, 100, C_X],
            },
            "Approximator": {
                "shapes": [ps, C_X],
                "filter_spec": [25, 100, 500, 100, C_X],
            },
            "Discriminator": {
                "shapes": [ps, C_X],
                "filter_spec": [25, 100, 200, 50, 1],
            },
        }
    else:
        TRANSLATION_SPEC = {
            "Generator": {"shapes": [ps, C_Y], "filter_spec": [25, C_X],},
            "Approximator": {"shapes": [ps, C_X], "filter_spec": [25, C_X],},
            "Discriminator": {"shapes": [ps, C_X], "filter_spec": [25, 1]},
        }
    print("Change Detector Init")
    cd = CGAN(TRANSLATION_SPEC, **CONFIG)
    print("Training")
    training_time = 0
    for epoch in trange(CONFIG["epochs"]):
        CONFIG.update(epochs=1)
        dataset = [x, y, Pu]
        TRAIN = tf.data.Dataset.from_tensor_slices(tuple(dataset))
        TRAIN = TRAIN.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        tr_time, _ = cd.train(TRAIN, evaluation_dataset=EVALUATE, **CONFIG)
        if epoch > 10:
            for x_im, y_im, _ in EVALUATE.batch(1):
                Pu = cd._change_map(cd([x_im, y_im]))
            Pu = image_in_patches(Pu, ps)
            Pu = tf.reshape(Pu, [-1, Pu.shape[-1]])
            Pu = tf.round(tf.reduce_mean(tf.cast(Pu, dtype=tf.float32), axis=-1))

    cd.final_evaluate(EVALUATE, **CONFIG)
    final_kappa = cd.metrics_history["cohens kappa"][-1]
    timestamp = cd.timestamp
    epoch = cd.epoch.numpy()
    return final_kappa, epoch, training_time, timestamp


if __name__ == "__main__":
    test("Texas")
    test("California")
