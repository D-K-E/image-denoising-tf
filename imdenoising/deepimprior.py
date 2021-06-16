"""!
\file deepimprior.py Deep Image prior implementation

This file implements the deep image prior paper in tensorflow: 

Ulyanov, D., Vedaldi, A., Lempitsky, V., 2020. Deep Image Prior. Int J Comput Vis 128, 1867–1888. https://doi.org/10.1007/s11263-020-01303-4


x = f_{theta}(z)

x: image
z: code vector ? -> random tensor of the type: \f[ R^{C \times W \times H}\f]
z is a randomly initialized 3D tensor, [0, 0.1]
theta: parameters: weights and bias of the filters in the network

The neural network is interpreted as a parametrization of the function 
f_{theta}(z). By parametrization we mean that, by differentiating the function
we can drive it to target output.

f_{theta}: is a neural network.

Without the additional data, the network captures the following statistics
about the image x:

- local, translation invariant convolutions
- pixel neighborhood at multiple scales.

Now we define an image denoising problem as the following: \f[p(x|x_0)\f].
Here the x_0 is the noisy image and we try to obtain the original image from
it.
Rather than modeling the distribution explicitly we regard as an optimization
problem of the following type:
\f[x' = argmin_x(E(x; x_0) + R(x)\f]

- \f[R(x)\f]: regularizer term 
- x_0 is the low resolution/noisy occluded image
- E(x; x_0) is data term: L^2 norm, that is \f[x' - x_0\f] where x' is the
  generated image and the x_0 is the original noisy image

So minimizing data term, E(x;x_0) means minimizing mean squared error loss

\f[
theta' = argmin_{theta} E(f_{theta}(z); x_0 ) 
x' = f_{theta'}(z)
\f]

- theta': local minimizer obtained using an optimizer such as gradient descent,
adam, etc.
- x_0: noisy image

The hyper parameters provided by the paper:

\f[z \in R^{3 \times W \times H} ∼ U(0, \frac{1}{10})\f]
\f[n_u = n_d = [8, 16, 32, 64, 128]\f]
\f[k_u = k_d = [3, 3, 3, 3, 3]\f]
\f[n_s = [0, 0, 0, 4, 4]\f]
\f[k_s = [NA, NA, NA, 1, 1]\f]
\f[\sigma_p = 30\f]
\f[num_iter = 2400\f]
\f[LR = 0.01\f]
\f[upsampling = bilinear\f]
"""
from typing import List, Optional
from PIL import Image
import tensorflow as tf
import numpy as np
import argparse
import os


from utils.tensorutils import random_tensor
from utils.imutils import normalize_image
from utils.layerutils import conv2d, in2d, lerelu, up2d, ReflectPadding2D
from utils.layerutils import batch_norm_layer, in2d
from utils.imutils import save_image
from utils.imutils import map_array_to_range


def make_z(ishape: List[int]):
    """!
    make z from paper
    """
    if any([a < 1 for a in ishape]):
        raise ValueError("input shape can not have an element less than 1")
    arr = np.random.rand(*ishape)
    narr = map_array_to_range(arr, mnv=0.0, mxv=0.1)
    narr = np.expand_dims(narr, 0)
    return narr


#
def d_i(filter_downsampling: int, kernel_size_downsampling: int, index: int = 1):
    """!
    \brief reproducing d_i from page 19 figure 21 of article
    """
    kd = kernel_size_downsampling
    conv_2 = conv2d(
        nb_filter=filter_downsampling,
        ksize_x=kd,
        ksize_y=kd,
        stride=2,
        name="d_i_conv2d_" + str(filter_downsampling) + "_" + str(index),
        padding="same",
    )
    # output rows = (Input height + 0 + 0 - kernel height) / (stride height) + 1
    conv_1 = conv2d(
        nb_filter=filter_downsampling, ksize_x=kd, ksize_y=kd, stride=1, padding="same"
    )
    bn1 = batch_norm_layer(gamma=0.99)
    bn2 = batch_norm_layer(gamma=0.99)
    #
    return [
        # first
        conv_2,
        # ReflectPadding2D(padding=(2, 2)),
        bn1,
        lerelu(),
        # second
        conv_1,
        # ReflectPadding2D(padding=(1, 1)),
        bn2,
        lerelu(),
    ]


#
def s_i(filter_skip: int, kernel_size_skip: int, index: int = 1):
    """!
    \brief reproducing s_i from page 19 figure 21 of article
    """
    ns = filter_skip
    ks = kernel_size_skip
    conv_1 = conv2d(
        nb_filter=ns,
        ksize_x=ks,
        ksize_y=ks,
        stride=1,
        padding="same",
        name="s_i_conv2d_" + str(filter_skip) + "_" + str(index),
    )
    bn = batch_norm_layer(gamma=0.99)
    return [conv_1, bn, lerelu()]  # ReflectPadding2D(padding=(1, 1)),


#
def u_i(filter_upsampling: int, kernel_size_upsampling: int, index: int = 1):
    """!
    \brief reproducing u_i from page 19 figure 21 of article
    """
    nu = filter_upsampling
    ku = kernel_size_upsampling
    return [
        # first
        conv2d(
            nb_filter=nu,
            ksize_x=ku,
            ksize_y=ku,
            stride=1,
            padding="same",
            name="u_i_conv2d_" + str(nu) + "_" + str(index),
        ),
        # ReflectPadding2D(padding=(1, 1)),
        batch_norm_layer(gamma=0.99),
        lerelu(),
        # second
        conv2d(nb_filter=nu, ksize_x=ku, ksize_y=ku, stride=1, padding="same"),
        # ReflectPadding2D(),
        batch_norm_layer(gamma=0.99),
        lerelu(),
        # up sample
        up2d(size_x=2, size_y=2, interpolation="bilinear"),
    ]


#
def apply_layers(inlayer, lst: List[tf.keras.layers.Layer]):
    """!
    \brief apply layers consecutively
    """
    x = inlayer
    for layer in lst:
        x = layer(x)
    return x


class DeepImPriorTrainModel(tf.keras.Model):
    """!
    \brief custom training model for fine tuning training process
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.predicted_value = None

    def train_step(self, data):
        """!
        \brief training step with additive noise as per paper

        standard deviation value is taken from paper page 19
        """
        #

        if len(data) == 3:
            z, y_img, sample_weight = data
        else:
            sample_weight = None
            z, y_img = data
        #
        zshape = z.shape.as_list()
        for i in range(len(zshape)):
            if zshape[i] is None:
                zshape[i] = 1
        mean = 0
        sigma = 1.0 / 30
        noise = np.random.default_rng().normal(mean, sigma, size=zshape)
        z += noise

        #
        with tf.GradientTape() as tape:
            y_pred = self(z, training=True)
            # reshape the prediction to match the z
            zs = [s for s in zshape]
            zs[-1] = -1
            y_prediction = tf.reshape(y_pred, zs)

            self.predicted_value = (y_prediction, y_img, z)
            # y_prediction = y_pred
            loss = self.compiled_loss(
                y_img,
                y_prediction,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )
        # save prediction as ndarray to save it later on
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_img, y_prediction)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


#
class DeepImPriorImSaveCallback(tf.keras.callbacks.Callback):
    """!
    \brief Save image depending on the epoch
    """

    def __init__(
        self,
        impath: str,
        mpath: str,
        imshape: List[int],
        verbose_save: bool,
        period: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.impath = impath
        self.imshape = imshape
        self.model_path = mpath
        self.verbose_save = verbose_save
        self.period = period

    def save_model_to_path(self, epoch: int):
        """!
        save model to given path
        """
        mpath = self.model_path + "_" + str(epoch)
        tf.keras.models.save_model(
            self.model, mpath, overwrite=True, include_optimizer=True, save_traces=True
        )

    def on_epoch_end(self, epoch, logs=None):
        """!
        On total there should be 2400 epoch as per paper page 19
        """
        if epoch % self.period == 0:
            pred, orig, noise = self.model.predicted_value
            im = pred.numpy().reshape(*self.imshape)
            # save prediction
            imname = self.impath + "_" + str(epoch) + ".png"
            save_image(image=im, fname=imname)
            if self.model_path is not None:
                self.save_model_to_path(epoch=epoch)
            if self.verbose_save:
                orig = orig.numpy().reshape(*self.imshape)
                noise = noise.numpy().reshape(*self.imshape)
                # save original
                imname = self.impath + "_orig_" + str(epoch) + ".png"
                save_image(image=orig, fname=imname)
                # save noise
                imname = self.impath + "_noise_" + str(epoch) + ".png"
                save_image(image=noise, fname=imname)


class DeepImPriorManager:
    ""

    def __init__(
        self,
        noisy_image: Image,
        verbose: bool,
        period: int,
        learning_rate: float,
        epochs: int,
        out_folder: str,
        out_prefix: str,
        save_model_path: Optional[str] = None,
        load_model_path: Optional[str] = None,
        plot_path: str = "plot_model.png",
        summary_path: str = "model_summary.txt",
        optimizer: str = "adam",
    ):
        """!
        Deep Image Prior training manager
        """
        self.image_info = (
            noisy_image.height,
            noisy_image.width,
            len(noisy_image.split()),
        )
        self.noisy_imarr = np.array(noisy_image)
        #
        self.verbose = verbose
        self.period = period
        #
        self.learning_rate = learning_rate
        self.epochs = epochs
        #
        self.out_impath = os.path.join(out_folder, out_prefix)
        #
        self.save_model_path = save_model_path
        self.load_model_path = load_model_path
        #
        self.plot_path = os.path.join(out_folder, plot_path)
        self.summary_path = os.path.join(out_folder, summary_path)
        # optimizer
        self.optimizer = optimizer
        self._cback = None
        self._model = None

    @property
    def callback(self) -> DeepImPriorImSaveCallback:
        """!
        Prepare the DeepImPriorImSaveCallback
        """
        if self._cback is None:
            self._cback = DeepImPriorImSaveCallback(
                impath=self.out_impath,
                imshape=self.image_info,
                mpath=self.save_model_path,
                verbose_save=self.verbose,
                period=self.period,
            )
        return self._cback

    @property
    def model(self) -> DeepImPriorTrainModel:
        """!
        \brief create the model that follows the architecture of the paper
        """
        if self._model is None:
            self._model = self.prep_model()
        return self._model

    def prep_model(self) -> DeepImPriorTrainModel:
        """!
        \brief create the model that follows the architecture of the paper

        The model is taken from page 19 figure 21
        """
        #
        rows, cols, channels = self.image_info
        # input layer
        ilayer = in2d(nb_rows=rows, nb_cols=cols, nb_channels=channels)
        #
        # down sampling layer
        skips = []
        k_d_s = 3
        x = ilayer
        n_ds = [8, 16, 32, 64, 128]
        for i in range(len(n_ds)):
            n_d = n_ds[i]
            lst = d_i(filter_downsampling=n_d, kernel_size_downsampling=k_d_s, index=i)
            x = apply_layers(x, lst)
            skips.append(x)
        #
        # upsampling with skip connections
        # n_us = list(reversed([8, 16, 32, 64, 128]))
        n_us = [8, 16, 32, 64, 128]
        n_ss = [0, 0, 0, 4, 4]
        k_ss = [None, None, None, 1, 1]
        for i in range(len(n_us)):
            #
            k_s = k_ss[i]
            n_u = n_us[i]
            n_s = n_ss[i]
            #
            if k_s is not None:
                x_ = skips[i]
                lst_s = s_i(filter_skip=n_s, kernel_size_skip=k_s, index=i)
                x_ = apply_layers(x_, lst_s)

                # resize the upsampled tensor to the skip connection tensor
                # except for the concatenation axis which is the last axis
                shapelst = list(x_.shape[:-1])
                shapelst.append(-1)
                x1_ = tf.reshape(x, shape=shapelst)

                x = tf.keras.layers.Concatenate()([x1_, x_])
                #
                lst_u = u_i(
                    filter_upsampling=n_u, kernel_size_upsampling=k_d_s, index=i
                )
                x = apply_layers(x, lst_u)

            else:
                lst_u = u_i(
                    filter_upsampling=n_u, kernel_size_upsampling=k_d_s, index=i
                )
                x = apply_layers(x, lst_u)
        #
        # last
        # upsampling to match z shape
        last = up2d(size_x=4, size_y=2, interpolation="bilinear")
        x = last(x)
        return DeepImPriorTrainModel(inputs=ilayer, outputs=x)

    def choose_optimizer(self):
        "Choose an optimizer"
        optimizer = None
        if self.optimizer.lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "ftrl":
            optimizer = tf.keras.optimizers.Ftrl(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate=self.learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        return optimizer

    def compile_model(self, optimizer_weights=None):
        """!
        Compile model with loss and optimization
        learning rate is taken from the page 19
        """
        if optimizer_weights is None:
            optimizer = self.choose_optimizer()
            self.model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.MSE,
                metrics=["accuracy", "mae"],
                run_eagerly=True,
            )
        else:
            optimizer, weights = optimizer_weights
            self.model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.MSE,
                metrics=["accuracy", "mae"],
                run_eagerly=True,
            )
            self.model.set_weights(weights)

    def init_model(self, optimizer_weights=None):
        """!
        Initialize model
        """
        #
        self.compile_model(optimizer_weights=optimizer_weights)
        #
        if self.verbose:
            with open(self.summary_path, "w", encoding="utf-8") as f:
                self.model.summary(print_fn=lambda x: f.write(x + "\n"))
            tf.keras.utils.plot_model(
                self.model, to_file=self.plot_path, show_dtype=True, show_shapes=True
            )

    def fit_model(self, x_train: np.ndarray, y_train: np.ndarray):
        """!
        \brief fit model
        """
        self.model.fit(
            x=x_train,
            y=y_train,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=[self.callback],
        )

    def predict_model(self, data: np.ndarray):
        """!
        """
        pred = self.model.predict(data)
        im = pred.reshape(*self.noisy_imarr.shape)
        return im

    def save_image(self, image: np.ndarray):
        """!
        \brief Save image
        """
        imname = self.callback.impath + "_" + "result" + ".png"
        save_image(image=image, fname=imname)

    def make_z_train(self):
        """!
        \brief make z vector
        """
        return make_z(ishape=self.noisy_imarr.shape)

    def make_train_target(self):
        ""
        return self.noisy_imarr[np.newaxis, :]

    def run(self):
        """!
        \brief run model
        """
        if self.load_model_path is not None:
            model = tf.keras.models.load_model(self.load_model_path)
            weights = model.get_weights()
            optimizer = model.optimizer
            self.init_model(optimizer_weights=(optimizer, weights))
        else:
            self.init_model()
        #
        x_train = self.make_z_train()
        y_train = self.make_train_target()
        self.fit_model(x_train=x_train, y_train=y_train)

    def run_save(self):
        """!
        \brief run and save the model
        """
        self.run()
        pred = self.predict_model(data=self.noisy_imarr.copy())
        self.save_image(image=pred)


#
def make_parser():
    """!
    create the argument parser and other related functions for io
    """
    parser = argparse.ArgumentParser(
        description="""
Denoise a given image using deep image prior algorithm.

Beware of the following issues before proceeding with the usage of this script:
- Convolution based algorithms are sensible to image size. Please use a square
  image. Ex: 800x800, or 600x600.

- Image size significantly effects the training. Either make sure you have
  enough computation power, or adjust the image size appropriately.

- Lastly as with all the gradient based methods we are using a stable learning
  rate. Feel free to adjust it before the training phase. I am thinking of
  adding decay learning rate option in the future.
        """,
        usage="""
python deepimprior.py ./data/im03.png --outpath ./data/outimages/ --outprefix
out_denoised --verbose 1 --epochs 10000 --learning_rate 0.1 --save_model_path
./data/outmodels/model --period 20 --optimizer adam
""",
    )
    parser.add_argument("imagepath", help="path to the image")
    parser.add_argument("--outpath", help="path for saving outputs", default="./")
    parser.add_argument(
        "--outprefix",
        help="prefix that will be prepended to intermediate files",
        default="outimg",
    )
    parser.add_argument(
        "--epochs", help="number of training epochs", type=int, default=2400
    )
    parser.add_argument(
        "--verbose",
        help="verbose output during training",
        type=int,
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--learning_rate",
        help="learning rate for the optimizer",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--save_model_path",
        help="Save model to path at each period of epochs",
        default=None,
    )
    parser.add_argument(
        "--load_model_path",
        help="Load model from path to resume training",
        default=None,
    )
    parser.add_argument(
        "--period",
        help="Periodic activity number, saving images, models etc at the end of each period/epoch number",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--optimizer",
        help="Optimizer to be used in the training process",
        type=str,
        choices=[
            "adam",
            "adamax",
            "adadelta",
            "adagrad",
            "ftrl",
            "nadam",
            "rmsprop",
            "sgd",
        ],
        default="adam",
    )
    return parser


def main_fn():
    ""
    parser = make_parser()
    args = parser.parse_args()
    verbose = bool(args.verbose)
    epochs = args.epochs
    if epochs <= 0:
        raise ValueError("epochs must be bigger than 0")
    #
    noisy_image = Image.open(args.imagepath)
    manager = DeepImPriorManager(
        noisy_image=noisy_image,
        verbose=verbose,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        period=args.period,
        out_folder=args.outpath,
        out_prefix=args.outprefix,
        save_model_path=args.save_model_path,
        load_model_path=args.load_model_path,
        optimizer=args.optimizer,
    )
    manager.run_save()


if __name__ == "__main__":
    main_fn()
