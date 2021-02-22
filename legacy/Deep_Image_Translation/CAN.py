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
import time
import argparse
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral
from pydensecrf.utils import create_pairwise_gaussian

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=-1, type=int)
parser.add_argument("--runs", default=1, type=int)
parser.add_argument("--run", type=int)
args = parser.parse_args()

LAMBDA = 5

TRAIN = 1
DATASET = args.dataset
NAME_DATASET = ["Texas", "California", "Shuguang"]
RUNS = args.runs
RUN = args.run

LEARNING_RATE = 10e-5
MAX_GRAD_NORM = 1.0
EPOCHS = 20
BATCH_SIZE = 100
PATCH_SIZE = 5
PATCH_STRIDE = 1


class CAN(object):
    def __init__(self, t1, t2, mask, folder):
        self.t1 = self.remove_borders(t1)
        self.t2 = self.remove_borders(t2)
        self.mask = self.remove_borders(mask)
        self.folder = folder
        self.shape_1 = self.t1.shape
        self.shape_2 = self.t2.shape
        size_y = [None, PATCH_SIZE, PATCH_SIZE, self.shape_1[-1]]
        size_x = [None, PATCH_SIZE, PATCH_SIZE, self.shape_2[-1]]
        self.x = tf.placeholder("float", size_x, name="x")  # t2
        self.y = tf.placeholder("float", size_y, name="y")  # t1
        self.Pu = tf.placeholder("float", [None, 1, 1, 1], name="Pu")
        self.G = self.G()
        self.A = self.A()
        self.D = self.D()
        self.CAN_Loss()

        self.model_name = "models/CAN/"
        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name)
        self.model_name += NAME_DATASET[DATASET] + ".ckpt"

    def G(self):
        with tf.name_scope("Gen") as sc:
            output1 = self.fully_con(self.y, 25, sc + "_1")
            output2 = self.fully_con(output1, 100, sc + "_2")
            output3 = self.fully_con(output2, 500, sc + "_3")
            output4 = self.fully_con(output3, 100, sc + "_4")
            output5 = self.fully_con(output4, self.shape_2[-1] * 25, sc + "_5")
            return tf.reshape(output5, [-1, 5, 5, self.shape_2[-1]])

    def A(self):
        with tf.name_scope("App") as sc:
            output1 = self.fully_con(self.x, 25, sc + "_1")
            output2 = self.fully_con(output1, 100, sc + "_2")
            output3 = self.fully_con(output2, 500, sc + "_3")
            output4 = self.fully_con(output3, 100, sc + "_4")
            output5 = self.fully_con(output4, self.shape_2[-1] * 25, sc + "_5")
            return tf.reshape(output5, [-1, 5, 5, self.shape_2[-1]])

    def D(self):
        with tf.name_scope("Dis") as sc:
            self.d = tf.concat([self.x, self.G], 0)
            output1 = self.fully_con(self.d, 25, sc + "_1")
            output2 = self.fully_con(output1, 100, sc + "_2")
            output3 = self.fully_con(output2, 200, sc + "_3")
            output4 = self.fully_con(output3, 50, sc + "_4")
            output5 = self.fully_con(output4, 1, sc + "_5", tf.nn.sigmoid)
            self.p1, self.p2 = tf.split(output5, 2, 0)

    def fully_con(self, Input, output_channel, scope, activation=tf.nn.leaky_relu):
        with tf.name_scope(scope):
            if len(Input.shape) > 2:
                kernel_shape = [Input.shape[-1] * PATCH_SIZE ** 2, output_channel]
            else:
                kernel_shape = [Input.shape[-1], output_channel]
            init = tf.contrib.layers.xavier_initializer()
            kernel = tf.get_variable(
                name=scope, shape=kernel_shape, initializer=init, dtype=tf.float32
            )
            bias = tf.Variable(
                tf.constant(0.0, shape=[output_channel], dtype=tf.float32)
            )
            outputs = tf.contrib.layers.flatten(Input)
            outputs = tf.nn.xw_plus_b(outputs, kernel, bias)
            if activation is not None:
                outputs = activation(outputs)
            return outputs

    def remove_borders(self, x):
        if PATCH_STRIDE != 1:
            s1 = x.shape[0]
            s2 = x.shape[1]
            remove1 = (s1 - PATCH_SIZE) % PATCH_STRIDE
            remove2 = (s2 - PATCH_SIZE) % PATCH_STRIDE
            r1u = remove1 // 2
            r1d = r1u - remove1
            if r1d == 0:
                r1d = None
            r2l = remove2 // 2
            r2r = r2l - remove2
            if r2r == 0:
                r2r = None
            x = x[r1u:r1d, r2l:r2r]
        return x

    def patches_from_idx(self, array, idx):
        res = []
        for k in range(idx.shape[0]):
            i = idx[k, 0]
            j = idx[k, 1]
            res.append(array[i : i + PATCH_SIZE, j : j + PATCH_SIZE, ...])
        return np.array(res)

    def idx_patches(self, A):
        i, j = 0, 0
        idx = []
        while i + PATCH_SIZE <= A.shape[0]:
            idx.append([i, j, 1])
            j += PATCH_STRIDE
            if j + PATCH_SIZE > A.shape[1]:
                i += PATCH_STRIDE
                j = 0
        return np.array(idx)

    def save_image(self, array, subfolder):
        img = Image.fromarray(array.astype("uint8"))
        if subfolder.find("Aff") != -1 or subfolder.find("d") != -1:
            img = ImageOps.equalize(img, mask=None)
        img = img.convert("RGB")
        img.save(self.folder + subfolder)

    def filtering(self, d):
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

    def CAN_Loss(self):
        with tf.name_scope("Losses"):
            self.loss_D = tf.reduce_mean(-(tf.log(self.p1 + 10e-10)) * self.Pu)
            self.loss_D += tf.reduce_mean(-(tf.log(1.0 - self.p2 + 10e-10)) * self.Pu)

            self.loss_G_d = tf.reduce_mean(tf.log(self.p1 + 10e-10) * self.Pu)
            self.loss_G_d += tf.reduce_mean(tf.log(1.0 - self.p2 + 10e-10) * self.Pu)

            weights = tf.tile(self.Pu, [1, PATCH_SIZE, PATCH_SIZE, self.shape_2[-1]])
            self.loss_G_L1 = tf.losses.absolute_difference(
                labels=self.x, predictions=self.G, weights=weights
            )

            self.loss_A = tf.losses.absolute_difference(
                labels=self.G, predictions=self.A, weights=weights
            )

            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(
                LEARNING_RATE, global_step, 100000, 0.96, staircase=True
            )
            optimizer = tf.train.AdamOptimizer(lr)
            losses = [LAMBDA * self.loss_G_L1 + self.loss_G_d, self.loss_A, self.loss_D]
            var = [
                tf.trainable_variables("Gen"),
                tf.trainable_variables("App"),
                tf.trainable_variables("Dis"),
            ]
            grads = []
            for i, v in enumerate(var):
                grads += tf.gradients(losses[i], v)
            var = [item for sublist in var for item in sublist]
            clipped_grad, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
            self.update = optimizer.apply_gradients(zip(clipped_grad, var), global_step)

        with tf.name_scope("Tensorboard"):
            if self.shape_1[-1] > 3:
                tf.summary.image(
                    "Y",
                    tf.cast((self.y[..., 1:4] + 1.0) / 2.0 * 255, tf.uint8),
                    max_outputs=10,
                )
            else:
                tf.summary.image(
                    "Y", tf.cast((self.y + 1.0) / 2.0 * 255, tf.uint8), max_outputs=10
                )
            if self.shape_2[-1] > 3:
                tf.summary.image(
                    "X",
                    tf.cast((self.x[..., 3:6] + 1.0) / 2.0 * 255, tf.uint8),
                    max_outputs=10,
                )
                tf.summary.image(
                    "G",
                    tf.cast((self.G[..., 3:6] + 1.0) / 2.0 * 255, tf.uint8),
                    max_outputs=10,
                )
                tf.summary.image(
                    "A",
                    tf.cast((self.A[..., 3:6] + 1.0) / 2.0 * 255, tf.uint8),
                    max_outputs=10,
                )
            else:
                tf.summary.image(
                    "X", tf.cast((self.x + 1.0) / 2.0 * 255, tf.uint8), max_outputs=10
                )
                tf.summary.image(
                    "G", tf.cast((self.G + 1.0) / 2.0 * 255, tf.uint8), max_outputs=10
                )
                tf.summary.image(
                    "A", tf.cast((self.A + 1.0) / 2.0 * 255, tf.uint8), max_outputs=10
                )
            weights = tf.tile(self.Pu, [1, PATCH_SIZE, PATCH_SIZE, 1])
            tf.summary.image("Pu", tf.cast(weights * 255, tf.uint8), max_outputs=10)
            tf.summary.scalar("Loss_Dis", self.loss_D)
            tf.summary.scalar("Loss_Gen_L1", self.loss_G_L1)
            tf.summary.scalar("Loss_Gen_d", self.loss_G_d)
            tf.summary.scalar("Loss_App", self.loss_A)

    def train_model(self):
        saver = tf.train.Saver()
        total_parameters = 0
        parameters = tf.trainable_variables()
        for variable in parameters:
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total parameters : {}".format(total_parameters))
        self.idx = self.idx_patches(self.mask)
        with tf.Session(config=config) as self.sess:
            writer_train = tf.summary.FileWriter("logs/train/CAN", self.sess.graph)
            merged = tf.summary.merge_all()
            self.sess.run(tf.global_variables_initializer())
            if TRAIN:
                tboard_epoch = 0
                num_batches = self.idx.shape[0] // BATCH_SIZE
                print("Number of batches per epoch: {}".format(num_batches))
                print("Leftovers: {}".format(self.idx.shape[0] % BATCH_SIZE))
                if self.idx.shape[0] % BATCH_SIZE != 0:
                    num_batches += 1
                try:
                    for epoch in tqdm(range(EPOCHS)):
                        idx_perm = np.random.permutation(self.idx.shape[0])
                        for p in tqdm(range(num_batches)):
                            temp_idx = self.idx[idx_perm[:BATCH_SIZE], :2]
                            Pu = self.idx[idx_perm[:BATCH_SIZE], 2]
                            Pu = Pu.reshape([-1, 1, 1, 1])
                            batch_t1 = self.patches_from_idx(self.t1, temp_idx)
                            batch_t2 = self.patches_from_idx(self.t2, temp_idx)
                            ### ADD NOISE
                            idx_perm = idx_perm[BATCH_SIZE:]
                            summary, _ = self.sess.run(
                                [merged, self.update],
                                {self.y: batch_t1, self.x: batch_t2, self.Pu: Pu},
                            )
                            if p % (num_batches // 5) == 0 and p > 0:
                                writer_train.add_summary(summary, tboard_epoch)
                                tboard_epoch += 1

                        self.evaluate(save=False)

                except KeyboardInterrupt:
                    print("\nTraining interrupted")
                saver.save(self.sess, self.model_name)
            else:
                saver.restore(self.sess, self.model_name)
            self.evaluate(save=True)

    def evaluate(self, save):
        mask = self.mask
        idx = np.copy(self.idx)
        x_hat = np.zeros_like(self.t2)
        y_hat = np.zeros_like(self.t2)
        runs = (idx.shape[0] // BATCH_SIZE) + (idx.shape[0] % BATCH_SIZE != 0)
        done_runs = 0
        diffs = []
        img_Pu = np.zeros(mask.shape)
        covers = np.zeros(mask.shape)
        for i in tqdm(range(runs)):
            temp_idx = idx[:BATCH_SIZE]
            batch_t1 = self.patches_from_idx(self.t1, temp_idx)
            batch_t2 = self.patches_from_idx(self.t2, temp_idx)
            tempG, tempA = self.sess.run(
                [self.G, self.A], {self.x: batch_t2, self.y: batch_t1}
            )
            if len(idx) >= BATCH_SIZE:
                bs = BATCH_SIZE
            else:
                bs = len(idx)
            l2sq = np.square(
                np.linalg.norm(tempG.reshape((bs, -1)) - tempA.reshape((bs, -1)), 1, 1)
            )
            diffs = np.append(diffs, l2sq, 0)
            for i in range(temp_idx.shape[0]):
                p = temp_idx[i, 0]
                q = p + PATCH_SIZE
                r = temp_idx[i, 1]
                s = r + PATCH_SIZE
                covers[p:q, r:s] += 1
                y_hat[p:q, r:s, :] += tempG[i, ...].reshape(
                    (PATCH_SIZE, PATCH_SIZE, self.shape_2[-1])
                )
                x_hat[p:q, r:s, :] += tempA[i, ...].reshape(
                    (PATCH_SIZE, PATCH_SIZE, self.shape_2[-1])
                )
                img_Pu[p:q, r:s] += np.tile(temp_idx[i, 2], (PATCH_SIZE, PATCH_SIZE))
            del tempG, tempA, temp_idx
            idx = idx[BATCH_SIZE:, ...]

        x_hat = np.divide(x_hat, covers[..., np.newaxis])
        y_hat = np.divide(y_hat, covers[..., np.newaxis])
        img_Pu = np.divide(img_Pu, covers)
        otsu = threshold_otsu(diffs)
        self.idx[:, -1] = (np.sign(otsu - diffs) + 1) / 2

        d = x_hat - y_hat
        d = np.linalg.norm(d, 1, 2)
        d[d > np.mean(d) + 3.0 * np.std(d)] = np.mean(d) + 3.0 * np.std(d)
        d = d / np.max(d)
        folder = self.folder

        heatmap = self.filtering(d)
        otsu = threshold_otsu(heatmap)
        CD_map = heatmap >= otsu

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
            Conf_map = np.zeros_like(CD_map)
            Conf_map = np.tile(Conf_map[..., np.newaxis], (1, 1, 3))
            index = np.logical_and(mask, CD_map)
            Conf_map[index] = [1, 1, 1]
            Conf_map[np.logical_and(mask, np.logical_not(CD_map)), :] = [1, 0, 0]
            Conf_map[np.logical_and(np.logical_not(mask), CD_map), :] = [0, 1, 0]

            self.save_image(255.0 * d, "d.png")
            self.save_image(255.0 * heatmap, "d_filtered.png")
            self.save_image(255.0 * Conf_map, "Confusion_map.png")
            self.save_image(255.0 * img_Pu, "1 - Pu.png")
            if self.t1.shape[-1] > 3:
                self.save_image(255.0 * (self.t1[..., 1:4] + 1.0) / 2.0, "t1_y.png")
            else:
                self.save_image(255.0 * (np.squeeze(self.t1) + 1.0) / 2.0, "t1_y.png")
            if self.t2.shape[-1] > 3:
                self.save_image(255.0 * (self.t2[..., 3:6] + 1.0) / 2.0, "t2_x.png")
                self.save_image(255.0 * (y_hat[..., 3:6] + 1.0) / 2.0, "t1_y_hat.png")
                self.save_image(255.0 * (x_hat[..., 3:6] + 1.0) / 2.0, "t2_x_hat.png")
            else:
                self.save_image(255.0 * (np.squeeze(self.t2) + 1.0) / 2.0, "t2_x.png")
                self.save_image(255.0 * (np.squeeze(y_hat) + 1.0) / 2.0, "t1_y_hat.png")
                self.save_image(255.0 * (np.squeeze(x_hat) + 1.0) / 2.0, "t2_x_hat.png")


def run_model():

    if DATASET == 1:
        mat = scipy.io.loadmat("data/California/UiT_HCD_California_2017.mat")
        t1 = np.array(mat["t1_L8_clipped"], dtype=float)
        t2 = np.array(mat["logt2_clipped"], dtype=float)
        t1 = block_reduce(t1, (4, 4, 1), np.mean)
        t2 = block_reduce(t2, (4, 4, 1), np.mean)
        mask = np.array(mat["ROI"], dtype=bool)
        mask = block_reduce(mask, (4, 4), np.max)
        folder = "Results/CAN/California/"
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
        folder = "Results/CAN/Texas/"
    elif DATASET == 2:
        mat = scipy.io.loadmat("data/Shuguang/shuguang_dataset.mat")
        t1 = np.array(mat["t1"], dtype=float)[:, :, 0]
        t2 = np.array(mat["t2"], dtype=float)
        mask = np.array(mat["ROI"], dtype=bool)
        t1 = t1 * 2.0 - 1.0
        t1 = t1[:, :, np.newaxis]
        t2 = t2 * 2.0 - 1.0
        temp = t2
        t2 = t1
        t1 = temp
        del temp
        folder = "Results/CAN/Shuguang/"
    else:
        print("Wrong data set")
        exit()
    del mat
    if not os.path.exists(folder):
        os.makedirs(folder)

    time1 = time.time()
    can = CAN(t1, t2, mask, folder)
    can.train_model()
    return can.evaluation, time.time() - time1


if __name__ == "__main__":
    evaluated, times = run_model()
    tf.reset_default_graph()
    os.system("rm -rf logs/*")
    time.sleep(2)
    print(*evaluated, times, sep=", ")
