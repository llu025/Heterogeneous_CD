import os

# Set loglevel to suppress tensorflow GPU messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
from itertools import count

import numpy as np
import tensorflow as tf
from scipy.io import loadmat, savemat
from change_priors import eval_prior, remove_borders, image_in_patches


def load_prior(name, expected_hw=None):
    """ Load prior from disk, validate dimensions and force shape to (h, w, 1) """
    mat = loadmat("data/" + name + "/change-prior.mat")
    varname = (
        re.sub(r"\W+", "", "aff" + str(expected_hw))
        if expected_hw is not None
        else "aff"
    )
    prior = tf.convert_to_tensor(mat[varname], dtype=tf.float32)
    if expected_hw is not None and prior.shape[:2] != expected_hw:
        raise FileNotFoundError
    if prior.ndim == 2:
        prior = prior[..., np.newaxis]
    return prior


def evaluate_prior(name, x, y, **kwargs):
    alpha = eval_prior(name, x, y, **kwargs)
    varname = re.sub(r"\W+", "", "aff" + str(x.shape[:2]))
    prior_path = "data/" + name + "/change-prior.mat"
    try:
        mat = loadmat(prior_path)
        mat.update({varname: alpha})
    except FileNotFoundError as e:
        mat = {varname: alpha}
    savemat(prior_path, mat)
    return alpha


def _denmark(reduce=False):
    """ Load Denmark dataset from .mat """
    mat = loadmat("data/Denmark/EMISAR_Foulum_PolSAR_logIntensity_CLband.mat")

    t1 = np.array(mat["imgCx"], dtype=np.single)
    t2 = np.array(mat["imgLy"], dtype=np.single)
    t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(mat["GT"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask


def _uk(reduce=True):
    """ Load UK dataset from .mat """
    mat = loadmat("data/UK/UK.mat")

    t1 = np.array(mat["t1"], dtype=np.single)
    t2 = np.array(mat["t2"], dtype=np.single)
    t1, t2 = _clip(t1), _clip(t2[..., np.newaxis])
    change_mask = tf.convert_to_tensor(mat["ROI"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reducing")
        reduction_ratios = (5, 5)
        new_dims = list(map(lambda a, b: a // b, change_mask.shape, reduction_ratios))
        t1 = tf.cast(tf.image.resize(t1, new_dims, antialias=True), dtype=tf.float32)
        t2 = tf.cast(tf.image.resize(t2, new_dims, antialias=True), dtype=tf.float32)
        change_mask = tf.cast(
            tf.image.resize(tf.cast(change_mask, tf.uint8), new_dims, antialias=True),
            tf.bool,
        )

    return t1, t2, change_mask


def _italy(reduce=False):
    """ Load Italy dataset from .mat """
    mat = loadmat("data/Italy/Italy.mat")

    t1 = np.array(mat["t1"], dtype=np.single)
    t2 = np.array(mat["t2"], dtype=np.single)
    if t1.shape[-1] == 3:
        t1 = t1[..., 0]
    t1, t2 = _clip(t1[..., np.newaxis]), _clip(t2)
    change_mask = tf.convert_to_tensor(mat["ROI"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask


def _france(reduce=True):
    """ Load France dataset from .mat """
    mat = loadmat("data/France/France.mat")

    t1 = np.array(mat["t1"], dtype=np.single)
    t2 = np.array(mat["t2"], dtype=np.single)
    t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(mat["ROI"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reducing")
        reduction_ratios = (5, 5)
        new_dims = list(map(lambda a, b: a // b, change_mask.shape, reduction_ratios))
        t1 = tf.cast(tf.image.resize(t1, new_dims, antialias=True), dtype=tf.float32)
        t2 = tf.cast(tf.image.resize(t2, new_dims, antialias=True), dtype=tf.float32)
        change_mask = tf.cast(
            tf.image.resize(tf.cast(change_mask, tf.uint8), new_dims, antialias=True),
            tf.bool,
        )

    return t1, t2, change_mask


def _california(reduce=False):
    """ Load California dataset from .mat """
    mat = loadmat("data/California/UiT_HCD_California_2017.mat")

    t1 = tf.convert_to_tensor(mat["t1_L8_clipped"], dtype=tf.float32)
    t2 = tf.convert_to_tensor(mat["logt2_clipped"], dtype=tf.float32)
    change_mask = tf.convert_to_tensor(mat["ROI"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reducing")
        reduction_ratios = (4, 4)
        new_dims = list(map(lambda a, b: a // b, change_mask.shape, reduction_ratios))
        t1 = tf.cast(tf.image.resize(t1, new_dims, antialias=True), dtype=tf.float32)
        t2 = tf.cast(tf.image.resize(t2, new_dims, antialias=True), dtype=tf.float32)
        change_mask = tf.cast(
            tf.image.resize(tf.cast(change_mask, tf.uint8), new_dims, antialias=True),
            tf.bool,
        )

    return t1, t2, change_mask


def _texas(clip=True):
    """ Load Texas dataset from .mat """
    mat = loadmat("data/Texas/Cross-sensor-Bastrop-data.mat")

    t1 = np.array(mat["t1_L5"], dtype=np.single)
    t2 = np.array(mat["t2_ALI"], dtype=np.single)
    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(mat["ROI_1"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]

    return t1, t2, change_mask


def _clip(image):
    """
        Normalize image from R_+ to [-1, 1].

        For each channel, clip any value larger than mu + 3sigma,
        where mu and sigma are the channel mean and standard deviation.
        Scale to [-1, 1] by (2*pixel value)/(max(channel)) - 1

        Input:
            image - (h, w, c) image array in R_+
        Output:
            image - (h, w, c) image array normalized within [-1, 1]
    """
    temp = np.reshape(image, (-1, image.shape[-1]))

    limits = tf.reduce_mean(temp, 0) + 3.0 * tf.math.reduce_std(temp, 0)
    for i, limit in enumerate(limits):
        channel = temp[:, i]
        channel = tf.clip_by_value(channel, 0, limit)
        ma, mi = tf.reduce_max(channel), tf.reduce_min(channel)
        channel = 2.0 * ((channel) / (ma)) - 1
        temp[:, i] = channel

    return tf.reshape(tf.convert_to_tensor(temp, dtype=tf.float32), image.shape)


def _training_data_generator(x, y, p, patch_size):
    """
        Factory for generator used to produce training dataset.
        The generator will choose a random patch and flip/rotate the images

        Input:
            x - tensor (h, w, c_x)
            y - tensor (h, w, c_y)
            p - tensor (h, w, 1)
            patch_size - int in [1, min(h,w)], the size of the square patches
                         that are extracted for each training sample.
        Output:
            to be used with tf.data.Dataset.from_generator():
                gen - generator callable yielding
                    x - tensor (ps, ps, c_x)
                    y - tensor (ps, ps, c_y)
                    p - tensor (ps, ps, 1)
                dtypes - tuple of tf.dtypes
                shapes - tuple of tf.TensorShape
    """
    c_x, c_y = x.shape[2], y.shape[2]
    chs = c_x + c_y + 1
    x_chs = slice(0, c_x, 1)
    y_chs = slice(c_x, c_x + c_y, 1)
    p_chs = slice(c_x + c_y, chs, 1)

    data = tf.concat([x, y, p], axis=-1)

    def gen():
        for _ in count():
            tmp = tf.image.random_crop(data, [patch_size, patch_size, chs])
            tmp = tf.image.rot90(tmp, np.random.randint(4))
            tmp = tf.image.random_flip_up_down(tmp)

            yield tmp[:, :, x_chs], tmp[:, :, y_chs], tmp[:, :, p_chs]

    dtypes = (tf.float32, tf.float32, tf.float32)
    shapes = (
        tf.TensorShape([patch_size, patch_size, c_x]),
        tf.TensorShape([patch_size, patch_size, c_y]),
        tf.TensorShape([patch_size, patch_size, 1]),
    )

    return gen, dtypes, shapes


DATASETS = {
    "Texas": _texas,
    "California": _california,
    "France": _france,
    "Italy": _italy,
    "UK": _uk,
    "Denmark": _denmark,
}
prepare_data = {
    "Texas": True,
    "California": True,
    "France": True,
    "Italy": False,
    "UK": True,
    "Denmark": False,
}


def fetch_fixed_dataset(name, patch_size=100, **kwargs):
    """
        Input:
            name - dataset name, should be in DATASETS
            kwargs - config {key: value} pairs.
                     Key should be in DATASET_DEFAULT_CONFIG
        Output:
            training_data - tf.data.Dataset with (x, y, prior)
                            shapes like (inf, patch_size, patch_size, ?)
            evaluation_data - tf.data.Dataset with (x, y, change_map)
                              shapes (1, h, w, ?)
            channels - tuple (c_x, c_y), number of channels for domains x and y
    """
    x_im, y_im, target_cm = DATASETS[name](prepare_data[name])

    try:
        initial_cm = load_prior(name, x_im.shape[:2])
    except (FileNotFoundError, KeyError) as e:
        print("Evaluating and saving prior")
        initial_cm = evaluate_prior(name, x_im, y_im, **kwargs)
    cross_loss_weight = 1 - initial_cm
    cross_loss_weight -= tf.reduce_min(cross_loss_weight)
    cross_loss_weight /= tf.reduce_max(cross_loss_weight)

    tr_gen, dtypes, shapes = _training_data_generator(
        x_im, y_im, cross_loss_weight, patch_size
    )
    training_data = tf.data.Dataset.from_generator(tr_gen, dtypes, shapes)
    training_data = training_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    dataset = [tf.expand_dims(tensor, 0) for tensor in [x_im, y_im, target_cm]]
    if not tf.config.list_physical_devices("GPU"):
        dataset = [tf.image.central_crop(tensor, 0.1) for tensor in dataset]
    evaluation_data = tf.data.Dataset.from_tensor_slices(tuple(dataset))

    c_x, c_y = shapes[0][-1], shapes[1][-1]

    return training_data, evaluation_data, (c_x, c_y)


def fetch_CGAN(name, **kwargs):
    """
        Input:
            name - dataset name, should be in DATASETS
            kwargs - config {key: value} pairs.
                     Key should be in DATASET_DEFAULT_CONFIG
        Output:
            training_data - tf.data.Dataset with (x, y, prior)
                            shapes like (inf, patch_size, patch_size, ?)
            evaluation_data - tf.data.Dataset with (x, y, change_map)
                              shapes (1, h, w, ?)
            channels - tuple (c_x, c_y), number of channels for domains x and y
    """
    ps = kwargs.get("patch_size")
    y_im, x_im, target_cm = DATASETS[name](prepare_data[name])
    if not tf.config.list_physical_devices("GPU"):
        dataset = [
            tf.image.central_crop(tensor, 0.1) for tensor in [x_im, y_im, target_cm]
        ]
    else:
        dataset = [x_im, y_im, target_cm]
    chs = [tensor.shape[-1] for tensor in dataset]
    dataset = [remove_borders(tensor, ps) for tensor in dataset]
    dataset = [tf.expand_dims(tensor, 0) for tensor in dataset]
    evaluation_data = tf.data.Dataset.from_tensor_slices(tuple(dataset))
    dataset = [image_in_patches(tensor, ps) for tensor in dataset]
    tot_patches = dataset[0].shape[0]
    return dataset[0], dataset[1], evaluation_data, (chs[0], chs[1]), tot_patches


def fetch(name, patch_size=100, **kwargs):
    """
        Input:
            name - dataset name, should be in DATASETS
            kwargs - config {key: value} pairs.
                     Key should be in DATASET_DEFAULT_CONFIG
        Output:
            training_data - tf.data.Dataset with (x, y, prior)
                            shapes like (inf, patch_size, patch_size, ?)
            evaluation_data - tf.data.Dataset with (x, y, change_map)
                              shapes (1, h, w, ?)
            channels - tuple (c_x, c_y), number of channels for domains x and y
    """
    x_im, y_im, target_cm = DATASETS[name](prepare_data[name])

    if not tf.config.list_physical_devices("GPU"):
        dataset = [
            tf.image.central_crop(tensor, 0.1) for tensor in [x_im, y_im, target_cm]
        ]
    else:
        dataset = [x_im, y_im, target_cm]

    dataset = [tf.expand_dims(tensor, 0) for tensor in dataset]
    x, y = dataset[0], dataset[1]
    evaluation_data = tf.data.Dataset.from_tensor_slices(tuple(dataset))

    c_x, c_y = x_im.shape[-1], y_im.shape[-1]

    return x, y, evaluation_data, (c_x, c_y)


if __name__ == "__main__":
    for DATASET in DATASETS:
        print(f"Loading {DATASET}")
        fetch_fixed_dataset(DATASET)
