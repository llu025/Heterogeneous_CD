# import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    UpSampling2D,
    Dense,
)
from tensorflow.keras.activations import relu, sigmoid, tanh
from tensorflow.keras.regularizers import l2
import tensorflow as tf


class ImageTranslationNetwork(Model):
    """
        Same as network in Luigis cycle_prior.

        Not supporting discriminator / Fully connected output.
        Support for this should be implemented as a separate class.
    """

    def __init__(
        self,
        input_chs,
        filter_spec,
        name,
        l2_lambda=1e-3,
        leaky_alpha=0.3,
        dropout_rate=0.2,
        dtype="float32",
    ):
        """
            Inputs:
                input_chs -         int, number of channels in input
                filter_spec -       list of integers specifying the filtercount
                                    for the respective layers
                name -              str, name of model
                leaky_alpha=0.3 -   float in [0,1], passed to the RELU
                                    activation of all but the last layer
                dropout_rate=0.2 -  float in [0,1], specifying the dropout
                                    probability in training time for all but
                                    the last layer
                dtype='float64' -   str or dtype, datatype of model
            Outputs:
                None
        """
        super().__init__(name=name, dtype=dtype)

        self.leaky_alpha = leaky_alpha
        self.dropout = Dropout(dropout_rate, dtype=dtype)
        conv_specs = {
            "kernel_size": 3,
            "strides": 1,
            "kernel_initializer": "GlorotNormal",
            "padding": "same",
            "kernel_regularizer": l2(l2_lambda),
            # "bias_regularizer": l2(l2_lambda),
            "dtype": dtype,
        }

        self.layers_ = []
        for l, n_filters in enumerate(filter_spec):  # OK
            if l == 0:
                layer = Conv2D(
                    n_filters,
                    input_shape=(None, None, input_chs),
                    name=f"{name}-{l:02d}",
                    **conv_specs,
                )
            else:
                layer = Conv2D(n_filters, name=f"{name}-{l:02d}", **conv_specs)
            self.layers_.append(layer)
            if "enc" in name:
                if l < len(filter_spec) // 2:
                    self.layers_.append(MaxPooling2D(name=f"{name}-MP_{l:02d}"))
                else:
                    if l < len(filter_spec) - 1:
                        self.layers_.append(UpSampling2D(name=f"{name}-UP_{l:02d}"))

    def call(self, x, training=False):
        skips = []
        for layer in self.layers_[:-1]:
            if "MP" in layer.name:
                skips.append(x)
                x = layer(x)
            elif "UP" in layer.name:
                x = layer(x)
                x = x + skips.pop()
            else:
                x = self.dropout(x, training)
                x = layer(x)
                x = relu(x, alpha=self.leaky_alpha)
        x = self.dropout(x, training)
        x = self.layers_[-1](x)
        x = tanh(x)
        return x


class Discriminator(Model):
    """
        CGAN by .. et. al discriminator
    """

    def __init__(
        self,
        shapes,
        filter_spec,
        name,
        l2_lambda=1e-3,
        leaky_alpha=0.3,
        dropout_rate=0.2,
        dtype="float32",
    ):
        """
            Inputs:
                input_chs -         int, number of channels in input
                filter_spec -       list of integers specifying the filtercount
                                    for the respective layers
                name -              str, name of model
                leaky_alpha=0.3 -   float in [0,1], passed to the RELU
                                    activation of all but the last layer
                dropout_rate=0.2 -  float in [0,1], specifying the dropout
                                    probability in training time for all but
                                    the last layer
                dtype='float64' -   str or dtype, datatype of model
            Outputs:
                None
        """
        super().__init__(name=name, dtype=dtype)
        self.leaky_alpha = leaky_alpha
        self.dropout = Dropout(dropout_rate, dtype=dtype)
        conv_specs = {
            "kernel_initializer": "GlorotNormal",
            # "kernel_regularizer": l2(l2_lambda),
            "dtype": dtype,
        }
        layer = Dense(
            filter_spec[0],
            input_shape=(None, shapes[0], shapes[0], shapes[1]),
            name=f"{name}-{0:02d}",
            **conv_specs,
        )
        self.layers_ = [layer]
        for l, n_filters in enumerate(filter_spec[1:]):
            layer = Dense(n_filters, name=f"{name}-{l+1:02d}", **conv_specs)
            self.layers_.append(layer)

    def call(self, inputs, training=False):
        """ Implements the feed forward part of the network """
        x = inputs
        for layer in self.layers_[:-1]:
            x = layer(x)
            x = relu(x, alpha=self.leaky_alpha)
            x = self.dropout(x, training)
        x = self.layers_[-1](x)
        return sigmoid(x)


class Generator(Model):
    """
        CGAN by .. et. al Generator and Approximator
    """

    def __init__(
        self,
        shapes,
        filter_spec,
        name,
        l2_lambda=1e-3,
        leaky_alpha=0.3,
        dropout_rate=0.2,
        dtype="float32",
    ):
        """
            Inputs:
                input_chs -         int, number of channels in input
                filter_spec -       list of integers specifying the filtercount
                                    for the respective layers
                name -              str, name of model
                leaky_alpha=0.3 -   float in [0,1], passed to the RELU
                                    activation of all but the last layer
                dropout_rate=0.2 -  float in [0,1], specifying the dropout
                                    probability in training time for all but
                                    the last layer
                dtype='float64' -   str or dtype, datatype of model
            Outputs:
                None
        """
        super().__init__(name=name, dtype=dtype)
        self.leaky_alpha = leaky_alpha
        self.dropout = Dropout(dropout_rate, dtype=dtype)
        self.ps = shapes[0]
        self.shape_out = filter_spec[-1]
        conv_specs = {
            "kernel_initializer": "GlorotNormal",
            # "kernel_regularizer": l2(l2_lambda),
            "dtype": dtype,
        }
        layer = Dense(
            filter_spec[0],
            input_shape=(None, self.ps, self.ps, shapes[1]),
            name=f"{name}-{0:02d}",
            **conv_specs,
        )
        self.layers_ = [layer]
        for l, n_filters in enumerate(filter_spec[1:]):
            layer = Dense(n_filters, name=f"{name}-{l+1:02d}", **conv_specs)
            self.layers_.append(layer)

    def call(self, inputs, training=False):
        """ Implements the feed forward part of the network """
        x = inputs
        for layer in self.layers_[:-1]:
            x = layer(x)
            x = relu(x, alpha=self.leaky_alpha)
            x = self.dropout(x, training)
        x = self.layers_[-1](x)
        x = relu(x, alpha=self.leaky_alpha)
        return tf.reshape(x, [-1, self.ps, self.ps, self.shape_out])


class CouplingNetwork(Model):
    """
        Same as network in Luigis cycle_prior.

        Not supporting discriminator / Fully connected output.
        Support for this should be implemented as a separate class.
    """

    def __init__(
        self,
        input_chs,
        filter_spec,
        name,
        decoder=False,
        l2_lambda=1e-3,
        dtype="float32",
    ):
        """
            Inputs:
                input_chs -         int, number of channels in input
                filter_spec -       list of integers specifying the filtercount
                                    for the respective layers
                name -              str, name of model
                leaky_alpha=0.3 -   float in [0,1], passed to the RELU
                                    activation of all but the last layer
                dropout_rate=0.2 -  float in [0,1], specifying the dropout
                                    probability in training time for all but
                                    the last layer
                dtype='float64' -   str or dtype, datatype of model
            Outputs:
                None
        """
        super().__init__(name=name, dtype=dtype)
        self.decoder = decoder
        conv_specs = {
            "kernel_size": 3,
            "strides": 1,
            "kernel_initializer": "GlorotNormal",
            "padding": "same",
            "kernel_regularizer": l2(l2_lambda),
            # "bias_regularizer": l2(l2_lambda),
            "dtype": dtype,
        }
        layer = Conv2D(
            filter_spec[0],
            input_shape=(None, None, input_chs),
            name=f"{name}-{0:02d}",
            **conv_specs,
        )
        self.layers_ = [layer]
        conv_specs.update(kernel_size=1)
        for l, n_filters in enumerate(filter_spec[1:]):
            layer = Conv2D(n_filters, name=f"{name}-{l+1:02d}", **conv_specs)
            self.layers_.append(layer)

    def call(self, inputs, training=False):
        """ Implements the feed forward part of the network """
        x = inputs
        for layer in self.layers_[:-1]:
            x = layer(x)
            x = sigmoid(x)
        x = self.layers_[-1](x)
        if self.decoder:
            x = tanh(x)
        else:
            x = sigmoid(x)
        return x
