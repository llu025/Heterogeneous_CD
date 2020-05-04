import tensorflow as tf
import tensorflow_addons as tfa
from decorators import image_to_tensorboard
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral
from pydensecrf.utils import create_pairwise_gaussian
import numpy as np


@tf.function
def threshold_otsu(image):
    """Return threshold value based on Otsu's method. Adapted to tf from sklearn
    Parameters
    ----------
    image : (N, M) ndarray
        Grayscale input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.
    Raises
    ------
    ValueError
         If ``image`` only contains a single grayscale value.
    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_otsu(image)
    >>> binary = image <= thresh
    Notes
    -----
    The input image must be grayscale.
    """
    if len(image.shape) > 2 and image.shape[-1] in (3, 4):
        msg = (
            "threshold_otsu is expected to work correctly only for "
            "grayscale images; image shape {0} looks like an RGB image"
        )
        warn(msg.format(image.shape))

    # Check if the image is multi-colored or not
    tf.debugging.assert_none_equal(
        tf.math.reduce_min(image),
        tf.math.reduce_max(image),
        summarize=1,
        message="expects more than one image value",
    )

    hist = tf.histogram_fixed_width(image, tf.constant([0, 255]), 256)
    hist = tf.cast(hist, tf.float32)
    bin_centers = tf.range(0.5, 256, dtype=tf.float32)

    # class probabilities for all possible thresholds
    weight1 = tf.cumsum(hist)
    weight2 = tf.cumsum(hist, reverse=True)
    # class means for all possible thresholds
    mean = tf.math.multiply(hist, bin_centers)
    mean1 = tf.math.divide(tf.cumsum(mean), weight1)
    # mean2 = (tf.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    mean2 = tf.math.divide(tf.cumsum(mean, reverse=True), weight2)

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    tmp1 = tf.math.multiply(weight1[:-1], weight2[1:])
    tmp2 = (mean1[:-1] - mean2[1:]) ** 2
    variance12 = tf.math.multiply(tmp1, tmp2)

    idx = tf.math.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


def _dense_gaussian_filtering(x, y, difference_img):
    """
        Concerning the filtering, the method proposed in
        krahenbuhl2011efficient is used. It exploits spatial context to
        filter $d$ with fully connected conditional random field models.
        It defines the pairwise edge potentials between all pairs of pixels
        in the image by a linear combination of Gaussian kernels in an
        arbitrary feature space. The main downside of the iterative
        optimisation of the random field lies in the fact that it requires
        the propagation of all the potentials across the image.
        However, this highly efficient algorithm reduces the computational
        complexity from quadratic to linear in the number of pixels by
        approximating the random field with a mean field whose iterative
        update can be computed using Gaussian filtering in the feature
        space. The number of iterations and the kernel width of the
        Gaussian kernels are the only hyper-parameters manually set,
        and we opted to tune them according to luppino2019unsupervised:
        $5$ iterations and a kernel width of $0.1$.
    """

    d = np.array(difference_img[0])
    d = np.concatenate((d, 1.0 - d), axis=2)
    W, H = d.shape[:2]
    stack = np.concatenate((x[0], y[0]), axis=-1)

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
    heatmap = np.array(Q, dtype=np.float32)
    heatmap = np.reshape(heatmap[0, ...], (1, W, H, 1))
    return tf.convert_to_tensor(heatmap)


def histogram_equalization(image):
    """
        I need a docstring, Luigi!
    """
    values_range = tf.constant([0.0, 255.0], dtype=tf.float32)
    histogram = tf.histogram_fixed_width(tf.cast(image, tf.float32), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]
    img_shape = tf.shape(image)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(
        tf.cast(cdf - cdf_min, tf.float32) * 255.0 / tf.cast(pix_cnt - 1, tf.float32)
    )
    px_map = tf.cast(px_map, tf.uint8)
    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
    return eq_hist


def decorated_median_filter(static_name, pre_process=histogram_equalization, **kwargs):
    """
        Wrap a tfa median filter with TensorBoard decorator and specify arguments
        Input:
            model - ChangeDetector, the model the filter is used with
            static_name - str, passed to decorators.image_to_tensorboard()
            pre_process - callable, passed to decorators.image_to_tensorboard()
            **kwargs - passed to tfa.image.median_filter2d
        Output:
            callable - takes input image as tfa.image.median_filter2d
    """

    @image_to_tensorboard(static_name=static_name)  # , pre_process=pre_process)
    def median_filter2d(self, x, y, difference_img):
        return tfa.image.median_filter2d(difference_img, **kwargs)

    return median_filter2d


def decorated_gaussian_filter(static_name, pre_process=histogram_equalization):
    """
        Wrap the gaussian filter with TensorBoard decorator and specify arguments
        Input:
            model - ChangeDetector, the model the filter is used with
            static_name - str, passed to decorators.image_to_tensorboard()
            pre_process - callable, passed to decorators.image_to_tensorboard()
        Output:
            callable - takes input image as tfa.image.median_filter2d
    """

    @image_to_tensorboard(static_name=static_name)  # , pre_process=pre_process)
    def gauss_filter(self, x, y, difference_img):
        return _dense_gaussian_filtering(x, y, difference_img)

    return gauss_filter


if __name__ == "__main__":
    import skimage.filters
    import numpy as np
    import matplotlib.pyplot as plt

    image = np.concatenate(
        (np.random.normal(0.2, 0.05, 100), np.random.normal(0.8, 0.05, 100))
    )
    image[image < 0], image[image > 1] = 0, 1

    ski_otsu_f = skimage.filters.threshold_otsu(image)

    image *= 255
    image.astype(int)

    ski_otsu_i = skimage.filters.threshold_otsu(image)
    with tf.device("cpu:0"):
        t = tf.convert_to_tensor(image, dtype=tf.int32)
        tf_otsu = threshold_otsu(t)

    print(f"ski float: {ski_otsu_f}")
    print(f"ski int:   {ski_otsu_i/255} ({ski_otsu_i})")
    print(f"tf:        {tf_otsu/255}    ({tf_otsu})")

    # plt.hist(image)
    # plt.savefig('/src/plots/hist.png')
