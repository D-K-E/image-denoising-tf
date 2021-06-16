"""!
\file layerutils.py functions for creation of layers
"""


import tensorflow as tf
from typing import Optional, Tuple, List

DATA_FORMAT = "channels_last"


def check_initializer(init: str):
    """!
    """
    cond1 = init == "glorot_uniform"
    cond2 = init == "glorot_normal"
    cond3 = init == "constant"
    cond4 = init == "he_normal"
    cond5 = init == "he_uniform"
    cond6 = init == "identity"
    cond7 = init == "lecun_normal"
    cond8 = init == "lecun_uniform"
    cond9 = init == "ones"
    cond10 = init == "orthogonal"
    cond11 = init == "random_normal"
    cond12 = init == "random_uniform"
    cond13 = init == "truncated_normal"
    cond14 = init == "variance_scaling"
    cond15 = init == "zeros"
    res = cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7
    res = res or cond8 or cond9 or cond10 or cond11 or cond12 or cond13
    res = res or cond14 or cond15
    return res


def in2d(
    nb_rows: int,
    nb_cols: int,
    nb_channels: int,
    batch_size: int = 1,
    dtype=tf.float32,
    name: Optional[str] = None,
):
    """!
    \brief Create an input layer to be consumed by the model

    \return tf.keras.layers.Input input layer of the model
    """
    if nb_rows < 1 or nb_cols < 1 or nb_channels < 1 or batch_size < 1:
        raise ValueError("input shape values must be bigger than 1")
    kwargs = {
        "shape": (int(nb_rows), int(nb_cols), int(nb_channels)),
        "dtype": dtype,
        "batch_size": batch_size,
    }
    if name is not None:
        return tf.keras.layers.Input(name=name, **kwargs)
    else:
        return tf.keras.layers.Input(**kwargs)


#
def lerelu(alpha=0.3, name: Optional[str] = None):
    """!
    \brief create a leaky relu layer
    \param alpha negative slope coefficient used in 
        \f[f(x) = \alpha * x if x < 0 \f]
    """
    if alpha == 1:
        raise ValueError("having 1 as value beats the purpose of using leaky relu")
    kwargs = {"alpha": float(alpha)}
    if name is not None:
        return tf.keras.layers.LeakyReLU(name=name, **kwargs)
    else:
        return tf.keras.layers.LeakyReLU(**kwargs)


#
def up2d(
    size_x: int = 2,
    size_y: int = 2,
    interpolation: str = "bilinear",
    name: Optional[str] = None,
):
    """!
    \brief create an upsampling layer

    \param size_x upsampling width size
    \param size_y upsampling height size
    \param interpolation interpolation method used in upsampling

    \return tf.Tensor with the shape (batch size, upsampled row, upsampled
        columns, channels)

    The layer expects that the following data format:
    (batch size, rows, cols, channels)
    """
    if size_x <= 0 or size_y <= 0:
        raise ValueError("size can not be lower than 1")
    if interpolation != "bilinear" and interpolation != "nearest":
        raise ValueError("interpolation must be bilinear or nearest")
    size = (int(size_x), int(size_y))
    kwargs = {"size": size, "data_format": DATA_FORMAT, "interpolation": interpolation}
    if name is not None:
        return tf.keras.layers.UpSampling2D(name=name, **kwargs)
    else:
        return tf.keras.layers.UpSampling2D(**kwargs)


#
def conv2d(
    nb_filter: int,
    ksize_x: int,
    ksize_y: int,
    stride: int,
    padding: str = "same",
    has_bias: bool = False,
    kernel_init: str = "glorot_uniform",
    bias_init: str = "glorot_uniform",
    name: Optional[str] = None,
):
    """!
    \brief create a 2d convolution layer

    pht: padding height top
    phb: padding height bottom
    pwt: padding width right
    pwb: padding width left

    sh: stride height
    ih: input height
    kh: kernel height
    sw: stride width
    iw: input width
    kw: kernel width

    Output height = (ih + pht + phb - kh) / (sh) + 1
    Output width = (iw + pwr + pwl - kw) / (sw) + 1

    output shape (batch size, output width, output height, filters)

    """
    if padding != "same" and padding != "valid":
        raise ValueError("padding must be same or valid: " + padding)
    if ksize_x % 2 != 1 and ksize_y % 2 != 1:
        raise ValueError("kernel size values must be an odd number")
    if check_initializer(kernel_init) is False:
        raise ValueError("unknown kernel initializer")
    if check_initializer(bias_init) is False:
        raise ValueError("unknown bias initializer")
    #
    ksize = (int(ksize_x), int(ksize_y))
    kwargs = {
        "filters": nb_filter,
        "kernel_size": ksize,
        "strides": stride,
        "padding": padding,
        "data_format": DATA_FORMAT,
        "use_bias": has_bias,
        "kernel_initializer": kernel_init,
        "bias_initializer": bias_init,
    }
    #
    if name is not None:
        return tf.keras.layers.Conv2D(name=name, **kwargs)
    else:
        return tf.keras.layers.Conv2D(**kwargs)


#
class ReflectPadding2D(tf.keras.layers.Layer):
    """!
    \brief reflect padding layer
    """

    def __init__(self, padding=(1, 1), **kwargs):
        super(ReflectPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)

    def call(self, arg):
        w_pad, h_pad = self.padding
        return tf.pad(arg, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], "REFLECT")


#
def batch_norm_layer(epsilon=0.001, momentum=0.99, gamma=1.0, **kwargs):
    """!
    \brief batch normalization layer
    """
    axis = -1 if DATA_FORMAT == "channels_last" else 1
    return tf.keras.layers.BatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        gamma_initializer="ones",
        scale=True,
        **kwargs
    )
