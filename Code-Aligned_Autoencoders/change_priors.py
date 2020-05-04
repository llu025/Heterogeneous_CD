import numpy as np
import tensorflow as tf
from decorators import write_image_to_png
from tqdm import trange
import re


def image_in_patches(x, ps):
    tmp = tf.cast(x, dtype=tf.float32)
    sizes = [1, ps, ps, 1]
    strides = [1, ps, ps, 1]
    rates = [1, 1, 1, 1]
    padding = "VALID"
    retval = tf.image.extract_patches(tmp, sizes, strides, rates, padding)
    retval = tf.reshape(retval, [-1, ps, ps, x.shape[-1]])
    return tf.cast(retval, dtype=x.dtype)


def _border_trim_slice(n, ps, pstr):
    """
        Helper function for remove_borders. Produce slices to trim t_n pixels
        along an axis, t_n//2 at the beginning and t_n//2 + t_n%2 at the end.
    """

    t_n = (n - ps) % pstr
    t_start, t_end = t_n // 2, t_n // 2 + t_n % 2
    assert t_start < n - t_end
    return slice(t_start, n - t_end)


def remove_borders(x, ps, pstr=None):
    """
        Trim the border of the images to fit the affinity patch size/stride.

        For a stride=1, borders trim is not done because the entire image is
        covered by the affinity computation.
        For stride>1, t_h, t_w pixels are removed in order to center the
        affinity computation. floor(t_h)/2 rows are removed on the top and
        floor(t_h)/2 or floor(t_h)/2+1 rows are removed from the bottom.
        Likewise for t_w

        Input:
            x - image with (h, w, c) size.
            ps - int
            stride - int
    """
    if pstr == 1:
        return x
    if pstr == None:
        pstr = ps
    h, w = x.shape[:2]
    slice_h = _border_trim_slice(h, ps, pstr)
    slice_w = _border_trim_slice(w, ps, pstr)

    return x[slice_h, slice_w, ...]


def from_idx_to_patches(x, idx, ps):
    """
    Given a list of pixels locations (idx), it extracts patches of size (ps,ps) from the
    image array x
        Input:
            x   - float, array of [image_height, image_width, num_channels], Input image
            idx - int, array of [total_num_patches, 2], Pixels locations in the image.
            ps  - int, patch size
        Output:
            res - float, array of [total_num_patches, patch_size, patch_size], Extracted patches
    """
    res = []
    for k in range(idx.shape[0]):
        i = idx[k, 0]
        j = i + ps
        l = idx[k, 1]
        m = l + ps
        res.append(x[i:j, l:m, ...])
    return tf.convert_to_tensor(res)


def patch_indecies(i_max: int, j_max: int, ps: int, pstr=None):
    """
    Given the sizes i_max and j_max of an image, it extracts the top-left corner pixel
    location of all the patches of size (ps,ps) and distant "pstr"
    pixels away from each other. If pstr < ps, the patches are overlapping.
        Input:
            i_max, j_max - int, sizes of the image
            ps           - int, patch size
            pstr         - int, patch stride
        Output:
            idx - int, array of [total_num_patches, 2], pixels locations
    """
    if pstr is None:
        pstr = ps
    idx = []
    for i in range(0, i_max - ps + 1, pstr):
        for j in range(0, j_max - ps + 1, pstr):
            idx.append([i, j])
    return tf.convert_to_tensor(idx)


def affinity(x):
    """
    Compute the affinity matrices of the patches of contained in a batch.
    It first computes the distances between the datapoints within a patch.
    Then it finds the suitable kernel width for each patch.
    Finally, applies the RBF.
        Input:
            x - float, array of [batch_size, patch_size, patch_size, num_channels],
                Batch of patches from data domain x.
        Output:
            A - float, array of [batch_size, patch_size^2, patch_size^2], Affinity matrix
    """
    _, h, w, c = x.shape
    x_1 = tf.expand_dims(tf.reshape(x, [-1, h * w, c]), 2)
    x_2 = tf.expand_dims(tf.reshape(x, [-1, h * w, c]), 1)
    A = tf.norm(x_1 - x_2, axis=-1)
    krnl_width = tf.math.top_k(A, k=A.shape[-1]).values
    krnl_width = tf.reduce_mean(input_tensor=krnl_width[:, :, (h * w) // 4], axis=1)
    krnl_width = tf.reshape(krnl_width, (-1, 1, 1))
    A = tf.exp(-(tf.divide(A, krnl_width) ** 2))
    return A


def alpha(x, y):
    """
    Compute the alpha prior starting from corresponding patches of data organized
    in batches. It first computes the affinity matrices of the two batches and then
    it computes the mean over the rows of the element-wise difference between the
    affinity matrices.
        Input:
            x - float, array of [batch_size, patch_size, patch_size, num_channels_x],
                Batch of patches from data domain x.
            y - float, array of [batch_size, patch_size, patch_size, num_channels_y],
                Batch of patches from data domain y.
        Output:
            alpha - float, array of [batch_size, patch_size, patch_size], prior
    """
    Ax = affinity(x)
    Ay = affinity(y)
    ps = int(Ax.shape[1] ** (0.5))
    alpha = tf.reshape(tf.reduce_mean(tf.abs(Ax - Ay), axis=-1), [-1, ps, ps])
    return alpha


def Degree_matrix(x, y):
    """
    Compute the degree matrix starting from corresponding patches of data organized
    in batches. It first computes the affinity matrices of the two batches and then
    it computes the norm of the difference between the rows of Ax and the rows of Ay.
    Then it is normalized.
        Input:
            x - float, array of [batch_size, patch_size, patch_size, num_channels_x],
                Batch of patches from data domain x.
            y - float, array of [batch_size, patch_size, patch_size, num_channels_y],
                Batch of patches from data domain y.
        Output:
            D - float, array of [batch_size, patch_size^2, patch_size^2], Degree matrix
    """
    ax = affinity(x)
    ay = affinity(y)
    D = tf.norm(tf.expand_dims(ax, 1) - tf.expand_dims(ay, 2), 2, -1)
    D = (D - tf.reduce_min(D)) / (tf.reduce_max(D) - tf.reduce_min(D))
    return D


def ztz(x, y):
    """
    Compute the inner product between datapoints from corresponding patches of data
    organized in batches. Since x and y are data between the range [-1,1],
    it is normalized to be between the range [0,1] using max_norm.
        Input:
            x - float, array of [batch_size, patch_size, patch_size, num_channels],
                Batch of patches from data domain x.
            y - float, array of [batch_size, patch_size, patch_size, num_channels],
                Batch of patches from data domain y.
        Output:
            ztz - float, array of [batch_size, patch_size^2, patch_size^2], Inner product
    """
    max_norm = x.shape[-1]
    flat_shape = [x.shape[0], x.shape[1] ** 2, -1]
    x = tf.reshape(x, flat_shape)
    y = tf.reshape(y, flat_shape)
    ztz = (tf.keras.backend.batch_dot(y, x, -1) + max_norm) / (2 * max_norm)

    return ztz


def patched_alpha(x, y, ps, pstr, **kwargs):
    """
    Given two images, it evaluates the alpha prior using affinity matrices computed on
    patches of size (ps,ps) and distant pstr one from each other. It first finds the
    location of the patches, then it evaluates alpha over batches and stores them in
    "Alpha". Every time a pixel is covered by a patch, its cover counter in "cover"
    is incremented and its "Alpha" value updated. Finally, "Alpha" is scaled by "covers".
        Input:
            x    - float, array of [image_height, image_width, num_channels_x], input image
            y    - float, array of [image_height, image_width, num_channels_y], input image
            ps   - int, patch size
            pstr - int, patch stride
        Output:
            Alpha - float, array of [image_height, image_width], Alpha prior
    """

    bs = kwargs.get("affinity_batch_size", 500)
    assert x.shape[:2] == y.shape[:2]
    y_max, x_max = x.shape[:2]
    Alpha = np.zeros((y_max, x_max), dtype=np.float32)
    covers = np.zeros((y_max, x_max), dtype=np.float32)

    idx = patch_indecies(y_max, x_max, ps, pstr)

    runs = idx.shape[0] // bs
    print("Runs: {}".format(runs))
    print("Leftovers: {}".format(idx.shape[0] % bs))
    done_runs = 0
    if idx.shape[0] % bs != 0:
        runs += 1
    for i in trange(runs):
        temp_idx = idx[:bs]
        batch_t1 = from_idx_to_patches(x, temp_idx, ps)
        batch_t2 = from_idx_to_patches(y, temp_idx, ps)
        al = alpha(batch_t1, batch_t2)
        for i in range(temp_idx.shape[0]):
            p = temp_idx[i, 0]
            q = p + ps
            r = temp_idx[i, 1]
            s = r + ps
            Alpha[p:q, r:s] += al[i]
            covers[p:q, r:s] += 1
        idx = idx[bs:]
    Alpha = np.divide(Alpha, covers)
    return Alpha


def eval_prior(name, x, y, **kwargs):
    """
    Given two images, it evaluates the Alpha prior on different resizing of them.
    It saves the intermediate results and the final result into a png and returns Alpha
        Input:
            name - str, dataset name
            x    - float, array of [image_height, image_width, num_channels_x], input image
            y    - float, array of [image_height, image_width, num_channels_y], input image
        Output:
            Alpha - float, array of [image_height, image_width], Alpha prior
    """
    sizes = kwargs.get("sizes", [-1, 0, 1])
    new_dims = np.array(x.shape[:-1])
    hw = re.sub(r"\W+", "", str(new_dims)) + "/"
    D = np.zeros((x.shape[0], x.shape[1], len(sizes)), dtype=np.float32)
    ps = kwargs.get("affinity_patch_size", 20)
    pstr = kwargs.get("affinity_stride", 5)
    ch = 0
    for i in sizes:
        if i < 0:
            ps = ps // 2
            pstr = ps // 4
            x_res = x
            y_res = y
        else:
            ps = kwargs.get("affinity_patch_size", 20)
            new_dims = new_dims // 2
            x_res = tf.image.resize(x, new_dims, antialias=True)
            y_res = tf.image.resize(y, new_dims, antialias=True)
        d = patched_alpha(
            remove_borders(x_res, ps, pstr),
            remove_borders(y_res, ps, pstr),
            ps,
            pstr,
            **kwargs
        )
        d = tf.expand_dims(d, -1)
        d = tf.image.resize(d, [D.shape[0], D.shape[1]], antialias=True)
        write_image_to_png(d, "data/" + name + "/Affinities/" + hw + str(i) + ".png")
        D[..., ch] = d[..., 0]
        ch += 1
    D = np.mean(D, axis=-1, keepdims=True)
    write_image_to_png(D, "data/" + name + "/Affinities/" + hw + "avg.png")
    return D


if __name__ == "__main__":
    pass
