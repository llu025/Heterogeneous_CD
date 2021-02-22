import scipy.io
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from PIL import Image
from PIL import ImageOps
from skimage.measure import block_reduce
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from skimage.filters import threshold_otsu
import sklearn.metrics as mt
import itertools
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral
from pydensecrf.utils import create_pairwise_gaussian
import time
import argparse
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=0, type=int)
args = parser.parse_args()

L = 16
LAMBDA = 0.2

PRETRAIN = 1
DATASET = args.dataset
NAME_DATASET = ["Texas", "California", "Shuguang"]

LEARNING_RATE = 10e-4
MAX_GRAD_NORM = 1.0
EPOCH = 500
pre_EPOCH = 250


class SCCN(object):
    def __init__(self, img_X, img_Y, mask, folder):
        self.mask = mask
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.img_X = img_X[np.newaxis, :]
        self.img_Y = img_Y[np.newaxis, :]
        self.shape_1 = self.img_X.shape
        self.shape_2 = self.img_Y.shape

        self.Input_X = tf.placeholder(
            dtype=tf.float32, shape=self.shape_1, name="Input_X"
        )

        self.Input_Y = tf.placeholder(
            dtype=tf.float32, shape=self.shape_2, name="Input_Y"
        )

        self.P = tf.placeholder(
            shape=[1, self.shape_1[1], self.shape_1[2], 1], dtype=tf.float32, name="Pu"
        )

        self.E_x = self.E_X()
        self.E_y = self.E_Y()
        self.D_x = self.D_X()
        self.D_y = self.D_Y()
        self.pretrain()
        self.get_model()

        self.model_name = "models/SCCN/"
        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name)
        self.model_name += NAME_DATASET[DATASET] + ".ckpt"

    def E_X(self):
        with tf.name_scope("E_X") as sc:

            # X Conv Layer
            conv_outputs_X = self.conv_2d(
                inputs=self.Input_X,
                kernel_size=(3, 3),
                output_channel=20,
                scope=sc + "_c",
                data_format="NHWC",
                activation=tf.nn.sigmoid,
            )

            # X Coupling Layer
            couple_outputs_X1 = self.coupling(
                Input=conv_outputs_X,
                output_channel=20,
                scope=sc + "_1",
                activation=tf.nn.sigmoid,
            )

            couple_outputs_X2 = self.coupling(
                Input=couple_outputs_X1,
                output_channel=20,
                scope=sc + "_2",
                activation=tf.nn.sigmoid,
            )

            couple_outputs_X3 = self.coupling(
                Input=couple_outputs_X2,
                output_channel=20,
                scope=sc + "_3",
                activation=tf.nn.sigmoid,
            )
            return couple_outputs_X3

    def E_Y(self):
        with tf.name_scope("E_Y") as sc:

            # Y Conv Layer
            conv_outputs_Y = self.conv_2d(
                inputs=self.Input_Y,
                kernel_size=(3, 3),
                output_channel=20,
                scope=sc + "_c",
                data_format="NHWC",
                activation=tf.nn.sigmoid,
            )

            # Y Coupling Layer
            couple_outputs_Y1 = self.coupling(
                Input=conv_outputs_Y,
                output_channel=20,
                scope=sc + "_1",
                activation=tf.nn.sigmoid,
            )

            couple_outputs_Y2 = self.coupling(
                Input=couple_outputs_Y1,
                output_channel=20,
                scope=sc + "_2",
                activation=tf.nn.sigmoid,
            )

            couple_outputs_Y3 = self.coupling(
                Input=couple_outputs_Y2,
                output_channel=20,
                scope=sc + "_3",
                activation=tf.nn.sigmoid,
            )

            return couple_outputs_Y3

    def D_X(self):
        with tf.name_scope("D_X") as sc:
            # X Coupling Layer
            deconv_outputs_X = self.coupling(
                Input=self.E_x,
                output_channel=self.shape_1[-1],
                scope=sc + "_1",
                activation=tf.nn.tanh,
            )

            return deconv_outputs_X

    def D_Y(self):
        with tf.name_scope("D_Y") as sc:
            # Y Coupling Layer
            deconv_outputs_Y = self.coupling(
                Input=self.E_y,
                output_channel=self.shape_2[-1],
                scope=sc + "_1",
                activation=tf.nn.tanh,
            )

            return deconv_outputs_Y

    def conv_2d(
        self,
        inputs,
        kernel_size,
        output_channel,
        scope,
        padding="SAME",
        data_format="NHWC",
        activation=tf.nn.sigmoid,
    ):
        with tf.name_scope(scope):
            kernel_h, kernel_w = kernel_size
            if data_format == "NHWC":
                kernel_shape = [
                    kernel_h,
                    kernel_w,
                    inputs.get_shape()[-1].value,
                    output_channel,
                ]
            else:
                kernel_shape = [
                    kernel_h,
                    kernel_w,
                    inputs.get_shape()[1].value,
                    output_channel,
                ]
            init = tf.contrib.layers.xavier_initializer()
            kernel = tf.get_variable(
                name=scope, shape=kernel_shape, initializer=init, dtype=tf.float32
            )
            outputs = tf.nn.conv2d(
                input=inputs,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding=padding,
                data_format=data_format,
            )
            bias = tf.Variable(
                tf.constant(0.0, shape=[output_channel], dtype=tf.float32)
            )
            outputs = outputs + bias
            if activation is not None:
                outputs = activation(outputs)
            return outputs

    def coupling(
        self,
        Input,
        output_channel,
        scope,
        padding="SAME",
        data_format="NHWC",
        activation=tf.nn.sigmoid,
    ):
        with tf.name_scope(scope):
            if data_format == "NHWC":
                kernel_shape = [1, 1, Input.get_shape()[-1].value, output_channel]
            else:
                kernel_shape = [1, 1, Input.get_shape()[1].value, output_channel]
            init = tf.contrib.layers.xavier_initializer()
            kernel = tf.get_variable(
                name=scope, shape=kernel_shape, initializer=init, dtype=tf.float32
            )
            outputs = tf.nn.conv2d(
                input=Input,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding=padding,
                data_format=data_format,
            )
            bias = tf.Variable(
                tf.constant(0.0, shape=[output_channel], dtype=tf.float32)
            )
            outputs = outputs + bias
            if activation is not None:
                outputs = activation(outputs)
            return outputs

    def get_model(self):
        with tf.name_scope("SCCN_Loss"):
            self.Diff = tf.sqrt(
                tf.reduce_sum(
                    tf.square(tf.subtract(self.E_x, self.E_y)), axis=-1, keep_dims=True
                )
            )
            # self.Diff has shape (Batch, width, height, 1)

            self.Loss = tf.reduce_mean(
                tf.multiply(self.P, self.Diff)
            ) - LAMBDA * tf.reduce_mean(self.P)
            # self.Loss has shape (1)

    def pretrain(self):
        with tf.name_scope("Pretraining"):
            self.pretrain_loss = tf.losses.mean_squared_error(self.Input_X, self.D_x)
            self.pretrain_loss += tf.losses.mean_squared_error(self.Input_Y, self.D_y)

            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(
                10e-3, global_step, 100000, 0.96, staircase=True
            )
            optimizer = tf.train.AdamOptimizer(lr)
            self.parameters = tf.trainable_variables()
            gradients = tf.gradients(self.pretrain_loss, self.parameters)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRAD_NORM)
            self.pretrain_op = optimizer.apply_gradients(
                zip(clipped_gradients, self.parameters), global_step
            )

    def save_image(self, array, subfolder):
        img = Image.fromarray(array.astype("uint8"))
        if subfolder.find("Aff") != -1 or subfolder.find("d") != -1:
            img = ImageOps.equalize(img, mask=None)
        img = img.convert("RGB")
        img.save(self.folder + subfolder)

    def filtering(self, d):
        if len(d.shape) == 2:
            d = d[..., np.newaxis]
        d = np.concatenate((d, 1.0 - d), axis=2)
        W = np.size(d, 0)
        H = np.size(d, 1)
        stack = np.concatenate((self.img_X[0], self.img_Y[0]), axis=2)
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

    def train_model(self):
        np.random.seed(None)
        prob = np.random.rand(1, self.shape_1[1], self.shape_1[2], 1)
        saver = tf.train.Saver()

        with tf.name_scope("Training"):
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(
                LEARNING_RATE, global_step, 100000, 0.96, staircase=True
            )
            var_list = tf.trainable_variables("E_X")
            optimizer = tf.train.AdamOptimizer(lr)
            gradients = tf.gradients(self.Loss, var_list)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRAD_NORM)
            self.train_op = optimizer.apply_gradients(
                zip(clipped_gradients, var_list), global_step
            )
        with tf.name_scope("Tensorboard"):
            tf.summary.image("H_X", tf.cast(self.E_x[..., 1:4] * 255, tf.uint8))
            tf.summary.image("H_Y", tf.cast(self.E_y[..., 1:4] * 255, tf.uint8))
            if self.shape_1[-1] > 3:
                tf.summary.image(
                    "D_X", tf.cast((self.D_x[..., 1:4] + 1.0) / 2.0 * 255, tf.uint8)
                )
                tf.summary.image(
                    "X", tf.cast((self.Input_X[..., 1:4] + 1.0) / 2.0 * 255, tf.uint8)
                )
            else:
                tf.summary.image(
                    "X", tf.cast((self.Input_X + 1.0) / 2.0 * 255, tf.uint8)
                )
                tf.summary.image("D_X", tf.cast((self.D_x + 1.0) / 2.0 * 255, tf.uint8))
            if self.shape_2[-1] > 3:
                tf.summary.image(
                    "Y", tf.cast((self.Input_Y[..., 3:6] + 1.0) / 2.0 * 255, tf.uint8)
                )
                tf.summary.image(
                    "D_Y", tf.cast((self.D_y[..., 3:6] + 1.0) / 2.0 * 255, tf.uint8)
                )
            else:
                tf.summary.image(
                    "Y", tf.cast((self.Input_Y + 1.0) / 2.0 * 255, tf.uint8)
                )
                tf.summary.image("D_Y", tf.cast((self.D_y + 1.0) / 2.0 * 255, tf.uint8))
            tf.summary.image("One minus Pu", tf.cast((1 - self.P) * 255, tf.uint8))
            tf.summary.image("Diff", tf.cast(self.Diff * 255, tf.uint8))
            tf.summary.scalar("Loss", self.Loss)
            tf.summary.scalar("Mean_diff", tf.reduce_mean(self.Diff))
            tf.summary.scalar("Pretrain_loss", self.pretrain_loss)

        with tf.Session(config=config) as sess:
            writer_pretr = tf.summary.FileWriter("logs/train/Sccn_pretr", sess.graph)
            writer_train = tf.summary.FileWriter("logs/train/Sccn", sess.graph)
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())

            if PRETRAIN:
                try:
                    for i in tqdm(range(pre_EPOCH)):
                        np.random.seed(None)
                        n_noise = np.random.normal(0, 0.05, self.shape_1)
                        # g_noise = np.random.gamma(L, 1/L, self.shape_2)
                        g_noise = np.random.normal(0, 0.05, self.shape_2)
                        summary, _ = sess.run(
                            [merged, self.pretrain_op],
                            feed_dict={
                                self.Input_X: self.img_X + n_noise,
                                self.Input_Y: self.img_Y + g_noise,
                                self.P: prob,
                            },
                        )

                        writer_pretr.add_summary(summary, i)
                        saver.save(sess, self.model_name)
                        del n_noise, g_noise, summary
                except KeyboardInterrupt:
                    print("\nPretraining interrupted")

            saver.restore(sess, self.model_name)
            """
            h_x, h_y = sess.run([self.E_x,self.E_y],
                                feed_dict={self.Input_X: self.img_X,
                                           self.Input_Y: self.img_Y})
            self.save_image(255.0*h_x[0,...,1:4],'H_X_Before.png')
            self.save_image(255.0*h_y[0,...,1:4],'H_Y_Before.png')
            """
            try:
                for _iter in tqdm(range(EPOCH)):
                    _, im, summary = sess.run(
                        [self.train_op, self.Diff, merged],
                        feed_dict={
                            self.Input_X: self.img_X,
                            self.Input_Y: self.img_Y,
                            self.P: prob,
                        },
                    )

                    writer_train.add_summary(summary, _iter)
                    if _iter > 10:
                        otsu = threshold_otsu(im)
                        prob = (np.sign(otsu - im) + 1) / 2
                    self.im = im
            except KeyboardInterrupt:
                print("\nTraining interrupted")
            self.evaluate(sess, save=True)

    def evaluate(self, session, save):
        mask = self.mask
        d, h_x, h_y = session.run(
            [self.Diff, self.E_x, self.E_y],
            feed_dict={self.Input_X: self.img_X, self.Input_Y: self.img_Y},
        )
        d = d[0]
        d[d > np.mean(d) + 3.0 * np.std(d)] = np.mean(d) + 3.0 * np.std(d)
        d = d / np.max(d)

        heatmap = self.filtering(d)
        otsu = threshold_otsu(heatmap)
        CD_map = heatmap >= otsu

        Confusion_map = np.zeros_like(CD_map)
        Confusion_map = np.tile(Confusion_map[..., np.newaxis], (1, 1, 3))
        Confusion_map[np.logical_and(mask, CD_map)] = [1, 1, 1]
        Confusion_map[np.logical_and(mask, np.logical_not(CD_map)), :] = [1, 0, 0]
        Confusion_map[np.logical_and(np.logical_not(mask), CD_map), :] = [0, 1, 0]

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
            if self.img_X.shape[-1] > 3:
                self.save_image(255.0 * (self.img_X[0, ..., 1:4] + 1.0) / 2.0, "x.png")
            else:
                self.save_image(255.0 * (np.squeeze(self.img_X) + 1.0) / 2.0, "x.png")
            if self.img_Y.shape[-1] > 3:
                self.save_image(255.0 * (self.img_Y[0, ..., 3:6] + 1.0) / 2.0, "y.png")
            else:
                self.save_image(255.0 * (np.squeeze(self.img_Y) + 1.0) / 2.0, "y.png")

            self.save_image(255.0 * heatmap, "d_filtered.png")
            self.save_image(255.0 * Confusion_map, "Confusion_map.png")
            self.save_image(255.0 * h_x[0, ..., 1:4], "H_X_After.png")
            self.save_image(255.0 * h_y[0, ..., 1:4], "H_Y_After.png")

        return d


def run_model(which_ch1=None, which_ch2=None):

    if DATASET == 1:
        mat = scipy.io.loadmat("data/California/UiT_HCD_California_2017.mat")
        t1 = np.array(mat["t1_L8_clipped"], dtype=float)
        t2 = np.array(mat["logt2_clipped"], dtype=float)
        t1 = block_reduce(t1, (4, 4, 1), np.mean)
        t2 = block_reduce(t2, (4, 4, 1), np.mean)
        mask = np.array(mat["ROI"], dtype=bool)
        mask = block_reduce(mask, (4, 4), np.max)
        folder = "Results/SCCN/California/"
    elif DATASET == 0:
        mat = scipy.io.loadmat("data/Texas/Cross-sensor-Bastrop-data.mat")
        mask = np.array(mat["ROI_1"], dtype=bool)
        t1 = np.array(mat["t1_L5"], dtype=float)
        t2 = np.array(mat["t2_ALI"], dtype=float)
        temp1 = np.reshape(t1, (-1, t1.shape[2]))
        temp2 = np.reshape(t2, (-1, t2.shape[2]))
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
        folder = "Results/SCCN/Texas/"
    elif DATASET == 2:
        mat = scipy.io.loadmat("data/Shuguang/shuguang_dataset.mat")
        t1 = np.array(mat["t1"], dtype=float)[:, :, 0]
        t2 = np.array(mat["t2"], dtype=float)
        mask = np.array(mat["ROI"], dtype=bool)
        t1 = t1 * 2.0 - 1.0
        t1 = t1[:, :, np.newaxis]
        t2 = t2 * 2.0 - 1.0
        folder = "Results/SCCN/Shuguang/"
    else:
        print("Wrong data set")
        exit()
    del mat

    time1 = time.time()
    sccn = SCCN(t1, t2, mask, folder)
    sccn.train_model()
    return sccn.evaluation, time.time() - time1


if __name__ == "__main__":
    evaluated, times = run_model()
    tf.reset_default_graph()
    os.system("rm -rf logs/*")
    time.sleep(2)
    print(*evaluated, times, sep=", ")
