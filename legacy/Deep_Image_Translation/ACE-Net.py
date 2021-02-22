import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import scipy.io
import os
import sys
import time
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from PIL import ImageOps
from PIL import Image
from skimage.filters import threshold_otsu
import sklearn.metrics as mt
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from skimage.measure import block_reduce
from scipy.ndimage import zoom
import argparse
import math
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral
from pydensecrf.utils import create_pairwise_gaussian

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=-1, type=int)
parser.add_argument("--recon", default=0.2, type=float)
parser.add_argument("--cycle", default=2.0, type=float)
parser.add_argument("--discr", default=1.0, type=float)
parser.add_argument("--tran", default=3.0, type=float)
parser.add_argument("--abl_alpha", default=False, type=bool)
args = parser.parse_args()

DATASET = args.dataset
PRE_TRAIN = 0
TRAIN = 1
NAME_DATASET = ["Texas", "California", "Shuguang"]
USE_PATCHES = 0  # DATASET

LEARNING_RATE = 10e-5
EPOCHS = 240
MAX_BATCHES = 10
BATCH_SIZE = 10
PATCH_SIZE = 100
PATCH_STRIDE = 5

PATCH_SIZE_AFF = 20
PATCH_STRIDE_AFF = PATCH_SIZE_AFF // 4
BATCH_SIZE_AFF = 200
ZERO_PAD = 0

W_REG = 0.001
W_RECON = args.recon
W_CYCLE = args.cycle
W_D = args.discr
W_HAT = args.tran
MAX_GRAD_NORM = 1.0
DROP_PROB = 0.2
ALPHA_LEAKY = 0.3

if DATASET == 1:
    nc1 = 11
    nc2 = 3
elif DATASET == 0:
    nc1 = 7
    nc2 = 10
elif DATASET == 2:
    nc1 = 1
    nc2 = 3
else:
    print("Wrong dataset")
    exit()

nf1 = 100
nf2 = 50
nf3 = 20
fs = 3

specs_enc_x = [[nc1, nf1, fs, 1], [nf1, nf2, fs, 1], [nf2, nf3, fs, 1]]

specs_dec_x = [[nf3, nf2, fs, 1], [nf2, nf1, fs, 1], [nf1, nc1, fs, 1]]

specs_enc_y = [[nc2, nf1, fs, 1], [nf1, nf2, fs, 1], [nf2, nf3, fs, 1]]

specs_dec_y = [[nf3, nf2, fs, 1], [nf2, nf1, fs, 1], [nf1, nc2, fs, 1]]

specs_disc = [[nf3, 64, 3, 1], [64, 32, 3, 1], [32, 16, 3, 1], [16, 1]]


class network(object):
    def __init__(self, specs, x=None, scope="Network", train=None):
        self.specs = specs
        self.scope = scope
        self.n_in = specs[0][0]
        self.temp = None
        if x is None:
            self.x = tf.placeholder(
                "float", [BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, self.n_in], name="X"
            )
        else:
            self.x = x
        if train is None:
            self.train = tf.placeholder(tf.bool, name="train")
        else:
            self.train = train
        with tf.name_scope("Parameters_Network"):
            with tf.name_scope("Weights"):
                self.weights = {}
                for i, l in enumerate(self.specs):
                    if len(l) > 2:
                        self.weights["layer_{}".format(i + 1)] = tf.Variable(
                            tf.truncated_normal(
                                [l[2], l[2], l[0], l[1]],
                                stddev=math.sqrt(2 / (l[2] * l[2] * l[0])),
                            ),
                            name="w_{}".format(i + 1),
                        )
                    if len(l) == 2:
                        flat = l[0] * PATCH_SIZE ** 2
                        self.weights["layer_{}".format(i + 1)] = tf.Variable(
                            tf.truncated_normal(
                                [flat, l[1]], stddev=math.sqrt(2 / (l[1] * flat))
                            ),
                            name="w_{}".format(i + 1),
                        )

            with tf.name_scope("Biases"):
                self.biases = {}
                for i, l in enumerate(self.specs):
                    if l[1] != 0:
                        self.biases["layer_{}".format(i + 1)] = tf.Variable(
                            tf.zeros(l[1]), name="b_{}".format(i + 1)
                        )

        self.layers()
        self.output = self.net()

    def train_var(self):
        val = [val for _, val in self.weights.items()]
        val += [val for _, val in self.biases.items()]
        return val

    def layers(self):
        last = len(self.specs)
        for i, l in enumerate(self.specs):
            if len(l) > 2:
                exec("self.layer_" + str(i + 1) + " = self.make_conv(l,i,last)")
            elif len(l) == 2:
                exec("self.layer_" + str(i + 1) + " = self.make_fully_con(l,i,last)")

    def net(self, x=None):
        with tf.name_scope(self.scope):
            if x is None:
                x = self.x
            for i, _ in enumerate(self.specs):
                if i == 0:
                    layer_input = x
                with tf.name_scope("layer_{}".format(i + 1)) as scope:
                    exec("self.temp = self.layer_" + str(i + 1) + "(layer_input)")
                layer_input = self.temp
            return self.temp

    def make_conv(self, l, i, last):
        def _function(layer_input):
            sc = "layer_{}".format(i + 1)
            out = tf.nn.conv2d(
                layer_input, self.weights[sc], [1, l[3], l[3], 1], padding="SAME"
            )
            out = tf.nn.bias_add(out, self.biases[sc])
            # out = tf.layers.batch_normalization(out,training = self.train)
            # out = batch_norm_layer(out,self.train)
            if i != (last - 1):
                out = self.leaky_relu(out, ALPHA_LEAKY)
                out = tf.layers.dropout(out, rate=DROP_PROB, training=self.train)
            else:
                out = tf.nn.tanh(out)
            return out

        return _function

    def make_fully_con(self, l, i, last):
        def _function(layer_input):
            sc = "layer_{}".format(i + 1)
            out = tf.matmul(tf.layers.flatten(layer_input), self.weights[sc])
            out = tf.nn.bias_add(out, self.biases[sc])
            # out = tf.layers.batch_normalization(out,training = self.train)
            # out = batch_norm_layer(out,self.train)
            if i != (last - 1):
                out = self.leaky_relu(out, ALPHA_LEAKY)
                out = tf.layers.dropout(out, rate=DROP_PROB, training=self.train)
            else:
                out = tf.nn.sigmoid(out)
            return out

        return _function

    def leaky_relu(self, features, alpha=0.3, name="LeakyRelu"):
        with tf.name_scope(name, "LeakyRelu", [features, alpha]):
            features = tf.convert_to_tensor(features, name="features")
            alpha = tf.convert_to_tensor(alpha, name="alpha")
            return tf.maximum(alpha * features, features)


class ACE(object):
    def __init__(self, t1, t2, mask):
        self.t1 = t1
        self.t2 = t2
        self.mask = mask
        self.model = "models/ACE-Net/"
        if not os.path.exists(self.model):
            os.makedirs(self.model)
        self.model += NAME_DATASET[DATASET] + ".ckpt"
        self.folder = "Results/ACE-Net/" + NAME_DATASET[DATASET] + "/"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        ### INPUT ###
        self.x = tf.placeholder("float", [None, None, None, nc1], name="x")
        self.y = tf.placeholder("float", [None, None, None, nc2], name="y")
        self.train = tf.placeholder(tf.bool, name="train")
        self.alpha = tf.placeholder("float", [None, None, None, 1], name="Prior")

        ### NETWORKS ###
        with tf.name_scope("X"):
            self.enc_x = network(specs_enc_x, self.x, "Encoder_x", self.train)
            self.code_x = self.enc_x.output
            self.dec_x = network(specs_dec_x, self.code_x, "Decoder_x", self.train)
            self.x_tilde = self.dec_x.output

        with tf.name_scope("Y"):
            self.enc_y = network(specs_enc_y, self.y, "Encoder_y", self.train)
            self.code_y = self.enc_y.output
            self.dec_y = network(specs_dec_y, self.code_y, "Decoder_y", self.train)
            self.y_tilde = self.dec_y.output

        with tf.name_scope("Discriminator"):
            self.d = tf.concat([self.code_x, self.code_y], 0)
            self.disc = network(specs_disc, scope="Discriminator", train=self.train)

        ### OUTPUT ###
        self.p1, self.p2 = tf.split(self.disc.net(self.d), 2, 0)
        self.y_hat = self.dec_y.net(self.code_x)
        self.x_cycle = self.dec_x.net(self.enc_y.net(self.y_hat))
        self.x_hat = self.dec_x.net(self.code_y)
        self.y_cycle = self.dec_y.net(self.enc_x.net(self.x_hat))

        ### LOSSES & OPTIMISATION ###
        with tf.name_scope("Losses"):
            with tf.name_scope("L2_loss"):
                reg_loss_x = 0
                for tf_var in self.enc_x.train_var() + self.dec_x.train_var():
                    reg_loss_x += tf.reduce_mean(tf.nn.l2_loss(tf_var))

                reg_loss_y = 0
                for tf_var in self.enc_y.train_var() + self.dec_y.train_var():
                    reg_loss_y += tf.reduce_mean(tf.nn.l2_loss(tf_var))

                reg_loss_d = 0
                for tf_var in self.disc.train_var():
                    reg_loss_d += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            self.loss_x_d = tf.reduce_mean(tf.squared_difference(0.0, self.p1))
            self.loss_x = tf.losses.mean_squared_error(self.x, self.x_tilde)
            self.loss_x_c = tf.losses.mean_squared_error(self.x, self.x_cycle)
            self.loss_y_hat = tf.losses.mean_squared_error(
                self.y, self.y_hat, 1.0 - self.alpha
            )

            self.loss_y_d = tf.reduce_mean(tf.squared_difference(1.0, self.p2))
            self.loss_y = tf.losses.mean_squared_error(self.y, self.y_tilde)
            self.loss_y_c = tf.losses.mean_squared_error(self.y, self.y_cycle)
            self.loss_x_hat = tf.losses.mean_squared_error(
                self.x, self.x_hat, 1.0 - self.alpha
            )
            self.tot_loss_ae = (
                W_D * (self.loss_y_d + self.loss_y_d)
                + W_CYCLE * (nc1 * self.loss_x_c + nc2 * self.loss_y_c) / (nc1 + nc2)
                + W_REG * (reg_loss_x + reg_loss_y)
                + W_RECON * (nc1 * self.loss_x + nc2 * self.loss_y) / (nc1 + nc2)
                + W_HAT * (nc1 * self.loss_y_hat + nc2 * self.loss_x_hat) / (nc1 + nc2)
            )

            self.loss_d = tf.reduce_mean(tf.squared_difference(1.0, self.p1))
            self.loss_d += tf.reduce_mean(tf.squared_difference(0.0, self.p2))
            self.tot_loss_d = self.loss_d + W_REG * reg_loss_d

            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(
                LEARNING_RATE, global_step, 100000, 0.96, staircase=True
            )
            optimizer = tf.train.AdamOptimizer(lr)
            losses = [self.tot_loss_ae, self.tot_loss_d]
            var = [
                tf.trainable_variables("X") + tf.trainable_variables("Y"),
                tf.trainable_variables("Discriminator"),
            ]
            grads = []
            for i, v in enumerate(var):
                grads += tf.gradients(losses[i], v)
            var = [item for sublist in var for item in sublist]
            clipped_grad, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
            self.update = optimizer.apply_gradients(zip(clipped_grad, var), global_step)

    def remove_borders(self, x):
        if PATCH_STRIDE_AFF != 1:
            s1 = x.shape[0]
            s2 = x.shape[1]
            remove_along_dim_1 = (s1 - PATCH_SIZE_AFF) % PATCH_STRIDE_AFF
            remove_along_dim_2 = (s2 - PATCH_SIZE_AFF) % PATCH_STRIDE_AFF
            up = remove_along_dim_1 // 2
            down = up - remove_along_dim_1
            if down == 0:
                down = None
            left = remove_along_dim_2 // 2
            right = left - remove_along_dim_2
            if right == 0:
                right = None
            x = x[up:down, left:right]
        return x

    def data_augmentation(self):
        batch_x = np.zeros([BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, nc1])
        batch_y = np.zeros([BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, nc2])
        batch_a = np.zeros([BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1])
        for i in range(BATCH_SIZE):
            rotation = np.random.randint(4)
            a = np.random.randint(self.t1.shape[0] - PATCH_SIZE)
            b = a + PATCH_SIZE
            c = np.random.randint(self.t1.shape[1] - PATCH_SIZE)
            d = c + PATCH_SIZE
            batch_x[i] = np.rot90(self.t1[a:b, c:d, :], rotation)
            batch_y[i] = np.rot90(self.t2[a:b, c:d, :], rotation)
            batch_a[i] = np.rot90(self.Alpha[a:b, c:d, np.newaxis], rotation)
            if np.random.randint(2):
                batch_x[i] = np.flipud(batch_x[i])
                batch_y[i] = np.flipud(batch_y[i])
                batch_a[i] = np.flipud(batch_a[i])

        return batch_x, batch_y, batch_a

    def idx_patches(self, array):
        i, j = 0, 0
        idx = []
        while i + PATCH_SIZE_AFF <= array.shape[0]:
            idx.append([i, j])
            j += PATCH_STRIDE_AFF
            if j + PATCH_SIZE_AFF > array.shape[1]:
                i += PATCH_STRIDE_AFF
                j = 0
        return np.array(idx)

    def from_idx_to_patches(self, array, idx):
        res = []
        if ZERO_PAD == 0:
            end = None
        else:
            end = -ZERO_PAD
        for k in range(idx.shape[0]):
            i = idx[k, 0]
            j = i + PATCH_SIZE_AFF
            l = idx[k, 1]
            m = l + PATCH_SIZE_AFF
            sz = PATCH_SIZE_AFF + 2 * ZERO_PAD
            padded_array = np.zeros((sz, sz) + array.shape[2:])
            padded_array[ZERO_PAD:end, ZERO_PAD:end, ...] = array[i:j, l:m, ...]
            res.append(padded_array)
        return np.array(res)

    def save_image(self, array, subfolder):
        img = Image.fromarray(array.astype("uint8"))
        if subfolder.find("Aff") != -1 or subfolder.find("d") != -1:
            img = ImageOps.equalize(img, mask=None)
        img = img.convert("RGB")
        img.save(self.folder + subfolder)

    def filtering(self, d):
        # print("Filtering!")
        d = d[..., np.newaxis]
        d = np.concatenate((d, 1.0 - d), axis=2)
        W = np.size(d, 0)
        H = np.size(d, 1)
        stack = np.concatenate((self.t1, self.t2), axis=2)
        CD = dcrf.DenseCRF2D(W, H, 2)
        d[d == 0] = 10e-20
        U = -(np.log(d))
        U = U.transpose(2, 0, 1).reshape((2, -1))
        U = U.copy(order="C")
        CD.setUnaryEnergy(U.astype(np.float32))
        pairwise_energy_gaussian = create_pairwise_gaussian((10, 10), (W, H))
        CD.addPairwiseEnergy(pairwise_energy_gaussian, compat=1)
        pairwise_energy_bilateral = create_pairwise_bilateral(
            sdims=(10, 10), schan=(0.1,), img=stack, chdim=2
        )
        CD.addPairwiseEnergy(pairwise_energy_bilateral, compat=1)
        Q = CD.inference(3)
        heatmap = np.array(Q)
        heatmap = np.reshape(heatmap[0, ...], (W, H))
        return heatmap

    def pre_train(self):
        D = np.zeros((self.t1.shape[0], self.t1.shape[1], 3))
        for i in range(-1, 2):
            if i < 0:
                x_resize = zoom(self.t1, (2 ** (-i), 2 ** (-i), 1))
                y_resize = zoom(self.t2, (2 ** (-i), 2 ** (-i), 1))
            else:
                x_resize = block_reduce(self.t1, (2 ** i, 2 ** i, 1), np.mean)
                y_resize = block_reduce(self.t2, (2 ** i, 2 ** i, 1), np.mean)
            print("Pre-training: Affinity on scale " + str(i))
            d = self.get_affinities(
                self.remove_borders(x_resize), self.remove_borders(y_resize)
            )
            d = np.array(Image.fromarray(d).resize((D.shape[0], D.shape[1])))
            self.save_image(255.0 * d, "Aff_" + str(i) + ".png")
            D[..., i] = d
        return np.mean(D, axis=-1)

    def get_affinities(self, x, y):
        aff = np.zeros((x.shape[0], x.shape[1]))
        covers = np.copy(aff)
        idx = self.idx_patches(aff)
        runs = idx.shape[0] // BATCH_SIZE_AFF
        if idx.shape[0] % BATCH_SIZE_AFF != 0:
            runs += 1
        for run in tqdm(range(runs)):
            temp_idx = idx[:BATCH_SIZE_AFF]
            batch_t1 = self.from_idx_to_patches(x, temp_idx)
            batch_t2 = self.from_idx_to_patches(y, temp_idx)
            Aff = self.sess.run(
                self.affinity(self.x, self.y), {self.x: batch_t1, self.y: batch_t2}
            )
            for i in range(temp_idx.shape[0]):
                p = temp_idx[i, 0]
                q = p + PATCH_SIZE_AFF
                r = temp_idx[i, 1]
                s = r + PATCH_SIZE_AFF
                aff[p:q, r:s] += Aff[i]
                covers[p:q, r:s] += 1
            idx = idx[BATCH_SIZE_AFF:]

        aff = np.divide(aff, covers)
        return aff

    def affinity(self, x, y):
        x_1 = tf.expand_dims(tf.reshape(x, [-1, PATCH_SIZE_AFF ** 2, nc1]), 2)
        x_2 = tf.expand_dims(tf.reshape(x, [-1, PATCH_SIZE_AFF ** 2, nc1]), 1)
        A_x = tf.norm(x_1 - x_2, axis=-1)
        del x_1, x_2
        k_x = tf.nn.top_k(A_x, k=A_x.shape[-1]).values
        k_x = tf.reduce_mean(k_x[:, :, -(PATCH_SIZE_AFF ** 2) // 2], axis=1)
        y_1 = tf.expand_dims(tf.reshape(y, [-1, PATCH_SIZE_AFF ** 2, nc2]), 2)
        y_2 = tf.expand_dims(tf.reshape(y, [-1, PATCH_SIZE_AFF ** 2, nc2]), 1)
        A_y = tf.norm(y_1 - y_2, axis=-1)
        del y_1, y_2
        k_y = tf.nn.top_k(A_y, k=A_y.shape[-1]).values
        k_y = tf.reduce_mean(k_y[:, :, -(PATCH_SIZE_AFF ** 2) // 2], axis=-1)
        k_x = tf.reshape(k_x, (-1, 1, 1))
        k_y = tf.reshape(k_y, (-1, 1, 1))
        A_x = tf.exp(-(tf.divide(A_x, k_x) ** 2))
        A_y = tf.exp(-(tf.divide(A_y, k_y) ** 2))
        D = tf.reshape(
            tf.reduce_mean(tf.abs(A_x - A_y), axis=-1),
            [-1, PATCH_SIZE_AFF, PATCH_SIZE_AFF],
        )
        return D

    def train_model(self):
        saver = tf.train.Saver()

        with tf.name_scope("Tensorboard"):
            tf.summary.image(
                "X_Recon",
                tf.cast((self.x_tilde[..., 1:4] + 1.0) / 2.0 * 255, tf.uint8),
                max_outputs=10,
            )
            tf.summary.image(
                "X_Cycle",
                tf.cast((self.x_cycle[..., 1:4] + 1.0) / 2.0 * 255, tf.uint8),
                max_outputs=10,
            )
            tf.summary.image(
                "X",
                tf.cast((self.x[..., 1:4] + 1.0) / 2.0 * 255, tf.uint8),
                max_outputs=10,
            )
            tf.summary.image(
                "Y_Cycle",
                tf.cast((self.y_cycle[..., 0:3] + 1.0) / 2.0 * 255, tf.uint8),
                max_outputs=10,
            )
            tf.summary.image(
                "Y_Recon",
                tf.cast((self.y_tilde[..., 0:3] + 1.0) / 2.0 * 255, tf.uint8),
                max_outputs=10,
            )
            tf.summary.image(
                "Y",
                tf.cast((self.y[..., 0:3] + 1.0) / 2.0 * 255, tf.uint8),
                max_outputs=10,
            )
            tf.summary.image(
                "X_hat",
                tf.cast((self.x_hat[..., 1:4] + 1.0) / 2.0 * 255, tf.uint8),
                max_outputs=10,
            )
            tf.summary.image(
                "Y_hat",
                tf.cast((self.y_hat[..., 0:3] + 1.0) / 2.0 * 255, tf.uint8),
                max_outputs=10,
            )
            tf.summary.image("Z", tf.cast(self.alpha * 255, tf.uint8), max_outputs=10)
            tf.summary.scalar("X_tr", self.loss_y_hat)
            tf.summary.scalar("X_Cycle", self.loss_x_c)
            tf.summary.scalar("X_Recon", self.loss_x)
            tf.summary.scalar("X_disc", self.loss_x_d)
            tf.summary.scalar("Y_tr", self.loss_x_hat)
            tf.summary.scalar("Y_Cycle", self.loss_y_c)
            tf.summary.scalar("Y_Recon", self.loss_y)
            tf.summary.scalar("Y_disc", self.loss_y_d)
            tf.summary.scalar("Z_disc", self.loss_d)

        with tf.Session(config=config) as self.sess:
            prior_path = "data/" + NAME_DATASET[DATASET] + "/change-prior.mat"
            prior_name = "aff" + str(self.mask.shape[0]) + str(self.mask.shape[1])
            try:
                if PRE_TRAIN:
                    raise Exception("Forcing prior computation")
                self.Alpha = np.squeeze(scipy.io.loadmat(prior_path)[prior_name])
            except Exception as exc:
                print(exc)
                print("Prior under evaluation")
                self.Alpha = self.pre_train()
                self.save_image(255.0 * self.Alpha, "Aff.png")
                scipy.io.savemat(prior_path, {prior_name: self.Alpha})

            if args.abl_alpha:
                self.Alpha = np.random.rand(*self.Alpha.shape)

            writer = tf.summary.FileWriter("logs/train/ACE-Net", graph=self.sess.graph)
            merged = tf.summary.merge_all()
            tf.global_variables_initializer().run()

            # Counting total number of paprameters
            total_parameters = 0
            parameters = tf.trainable_variables()
            for variable in parameters:
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            # print("Total number of parameters : {}".format(total_parameters))

            if TRAIN:
                try:
                    for epoch in tqdm(range(EPOCHS)):
                        for p in range(MAX_BATCHES):
                            b_1, b_2, b_a = self.data_augmentation()
                            summary, _ = self.sess.run(
                                [merged, self.update],
                                {
                                    self.x: b_1,
                                    self.y: b_2,
                                    self.alpha: b_a,
                                    self.train: True,
                                },
                            )
                        writer.add_summary(summary, epoch)
                        if epoch % (EPOCHS // 3) == 0 and epoch > 0:
                            self.Alpha = self.evaluate(save=False)
                    saver.save(self.sess, self.model)
                except KeyboardInterrupt:
                    print("\ntraining interrupted")
            else:
                saver.restore(self.sess, self.model)

            _ = self.evaluate(save=True)

    def transform_images(self):
        if USE_PATCHES:
            x_hat = np.zeros(self.t1.shape)
            y_hat = np.zeros(self.t2.shape)
            covers = np.zeros(self.mask.shape)
            idx = self.idx_patches(covers)
            runs = idx.shape[0] // BATCH_SIZE_AFF
            print("Runs: {}".format(runs))
            print("Leftovers: {}".format(idx.shape[0] % BATCH_SIZE_AFF))
            if self.idx.shape[0] % BATCH_SIZE_AFF != 0:
                runs += 1
            for i in tqdm(range(runs)):
                temp_idx = idx[0:BATCH_SIZE_AFF, :]
                batch_t1 = self.from_idx_to_patches(self.t1, temp_idx)
                batch_t2 = self.from_idx_to_patches(self.t2, temp_idx)
                temp2, temp1 = self.sess.run(
                    [self.x_hat, self.y_hat],
                    {self.x: batch_t1, self.y: batch_t2, self.train: False},
                )
                for i in range(temp_idx.shape[0]):
                    p = temp_idx[i, 0]
                    q = p + PATCH_SIZE_AFF
                    r = temp_idx[i, 1]
                    s = q + PATCH_SIZE_AFF
                    y_hat[p:q, r:s, :] += temp1[i]
                    x_hat[p:q, r:s, :] += temp2[i]
                    covers[p:q, r:s] += 1
                del temp_idx
                idx = idx[BATCH_SIZE_AFF:]

            x_hat = np.divide(x_hat, covers[..., np.newaxis])
            y_hat = np.divide(y_hat, covers[..., np.newaxis])
            time2 = time.time()
            print((time2 - time1) // 3600)
        else:
            t1 = self.t1[np.newaxis, ...]
            t2 = self.t2[np.newaxis, ...]
            x_hat, y_hat = self.sess.run(
                [self.x_hat, self.y_hat], {self.x: t1, self.y: t2, self.train: False}
            )
            y_hat = y_hat[0]
            x_hat = x_hat[0]

        return x_hat, y_hat

    def evaluate(self, save):
        x_hat, y_hat = self.transform_images()
        d_x = self.t1 - x_hat
        d_y = self.t2 - y_hat
        d_x = np.linalg.norm(d_x, 2, -1)
        d_y = np.linalg.norm(d_y, 2, -1)

        d_x[d_x > np.mean(d_x) + 3.0 * np.std(d_x)] = np.mean(d_x) + 3.0 * np.std(d_x)
        d_y[d_y > np.mean(d_y) + 3.0 * np.std(d_y)] = np.mean(d_y) + 3.0 * np.std(d_y)
        d_x = d_x / np.max(d_x)
        d_y = d_y / np.max(d_y)
        d = (d_x + d_y) / 2.0

        AUC_b = mt.roc_auc_score(self.mask.flatten(), d.flatten())
        otsu = threshold_otsu(d)
        CD_map = d >= otsu
        F1_Score_b = mt.f1_score(self.mask.flatten(), CD_map.flatten())
        OA_b = mt.accuracy_score(self.mask.flatten(), CD_map.flatten())
        KC_b = mt.cohen_kappa_score(self.mask.flatten(), CD_map.flatten())

        heatmap = self.filtering(d)
        otsu = threshold_otsu(heatmap)  # local_otsu = otsu(heatmap, disk(15))
        CD_map = heatmap >= otsu  # CD_map = heatmap >= local_otsu

        conf_map = np.zeros_like(CD_map)
        conf_map = np.tile(conf_map[..., np.newaxis], (1, 1, 3))
        conf_map[np.logical_and(self.mask, CD_map)] = [1, 1, 1]
        conf_map[np.logical_and(self.mask, np.logical_not(CD_map)), :] = [1, 0, 0]
        conf_map[np.logical_and(np.logical_not(self.mask), CD_map), :] = [0, 1, 0]

        AUC = mt.roc_auc_score(self.mask.flatten(), heatmap.flatten())
        AUPRC = mt.average_precision_score(self.mask.flatten(), heatmap.flatten())

        PREC_0 = mt.precision_score(self.mask.flatten(), CD_map.flatten(), pos_label=0)
        PREC_1 = mt.precision_score(self.mask.flatten(), CD_map.flatten())
        REC_0 = mt.recall_score(self.mask.flatten(), CD_map.flatten(), pos_label=0)
        REC_1 = mt.recall_score(self.mask.flatten(), CD_map.flatten())
        KC = mt.cohen_kappa_score(self.mask.flatten(), CD_map.flatten())
        [[TN, FP], [FN, TP]] = mt.confusion_matrix(
            self.mask.flatten(), CD_map.flatten()
        )
        self.evaluation = [TP, TN, FP, FN, PREC_0, REC_0, PREC_1, REC_1, KC, AUC, AUPRC]
        if save:
            self.save_image(255.0 * d_x, "d_x.png")
            self.save_image(255.0 * d_y, "d_y.png")
            self.save_image(255.0 * d, "d.png")
            self.save_image(255.0 * heatmap, "d_filtered.png")
            self.save_image(255.0 * conf_map, "Confusion_map.png")
            if nc1 > 3:
                self.save_image(255.0 * (self.t1[..., 1:4] + 1.0) / 2.0, "x.png")
                self.save_image(255.0 * (x_hat[..., 1:4] + 1.0) / 2.0, "x_hat.png")
            else:
                self.save_image(255.0 * (np.squeeze(self.t1) + 1.0) / 2.0, "x.png")
                self.save_image(255.0 * (np.squeeze(x_hat) + 1.0) / 2.0, "x_hat.png")
            if nc2 > 3:
                self.save_image(255.0 * (self.t2[..., 3:6] + 1.0) / 2.0, "y.png")
                self.save_image(255.0 * (y_hat[..., 3:6] + 1.0) / 2.0, "y_hat.png")
            else:
                self.save_image(255.0 * (np.squeeze(self.t2) + 1.0) / 2.0, "y.png")
                self.save_image(255.0 * (np.squeeze(y_hat) + 1.0) / 2.0, "y_hat.png")
        return d


def run_model():
    if DATASET == 1:
        mat = scipy.io.loadmat("data/California/UiT_HCD_California_2017.mat")
        t1 = np.array(mat["t1_L8_clipped"], dtype=float)
        t2 = np.array(mat["logt2_clipped"], dtype=float)
        mask = np.array(mat["ROI"], dtype=bool)
        t1 = block_reduce(t1, (4, 4, 1), np.mean)
        t2 = block_reduce(t2, (4, 4, 1), np.mean)
        mask = block_reduce(mask, (4, 4), np.max)
    elif DATASET == 0:
        mat = scipy.io.loadmat("data/Texas/Cross-sensor-Bastrop-data.mat")
        mask = np.array(mat["ROI_1"], dtype=bool)
        t1 = np.array(mat["t1_L5"], dtype=float)
        t2 = np.array(mat["t2_ALI"], dtype=float)
        temp1 = np.reshape(t1, (-1, nc1))
        temp2 = np.reshape(t2, (-1, nc2))
        limits = np.mean(temp1, 0) + 3.0 * np.std(temp1, 0)
        for channel, limit in enumerate(limits):
            temp = temp1[:, channel]
            temp[temp > limit] = limit
            temp = 2.0 * temp / np.max(temp) - 1.0
            temp1[:, channel] = temp
        limits = np.mean(temp2, 0) + 3.0 * np.std(temp2, 0)
        for channel, limit in enumerate(limits):
            temp = temp2[:, channel]
            temp[temp > limit] = limit
            temp = 2.0 * temp / np.max(temp) - 1.0
            temp2[:, channel] = temp
        t1 = np.reshape(temp1, np.shape(t1))
        t2 = np.reshape(temp2, np.shape(t2))
        del temp1, temp2, limits, temp
    elif DATASET == 2:
        mat = scipy.io.loadmat("data/Shuguang/shuguang_dataset.mat")
        t1 = np.array(mat["t1"], dtype=float)[:, :, 0]
        t2 = np.array(mat["t2"], dtype=float)
        mask = np.array(mat["ROI"], dtype=bool)
        t1 = t1 * 2.0 - 1.0
        t1 = t1[:, :, np.newaxis]
        t2 = t2 * 2.0 - 1.0
    else:
        print("Wrong data set")
        exit()
    del mat
    time1 = time.time()
    ace = ACE(t1, t2, mask)
    ace.train_model()
    return ace.evaluation, time.time() - time1


if __name__ == "__main__":
    evaluated, times = run_model()
    tf.reset_default_graph()
    # os.system("rm -rf /logs/*")
    time.sleep(2)
    print(*evaluated, times, sep=", ")
