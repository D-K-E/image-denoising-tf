"""!
\file imutils.py image utils
"""

from PIL import Image
import numpy as np


def map_array_to_range(arr: np.ndarray, mnv: float, mxv: float):
    """!
    \brief maps the array range to given interval
    """
    narr = arr / arr.max()
    return mnv + (mxv - mnv) * narr


def map_image_to_range(img: np.ndarray, mnv: float, mxv: float):
    """!
    \brief map image to a given range
    """
    nimg = img.copy()
    if img.ndim == 2:
        return map_array_to_range(nimg, mnv=mnv, mxv=mxv)
    if img.ndim == 3:
        for channel in range(img.shape[2]):
            nimg[:, :, channel] = map_array_to_range(
                nimg[:, :, channel], mnv=mnv, mxv=mxv
            )
        return nimg
    else:
        raise ValueError("number of dimensions must be 2/3 for image")


def normalize_image(img: np.ndarray):
    """!
    \brief normalize image
    """
    return img.astype(np.float) / 255.0


def save_image(image, fname: str):
    """!
    \brief save image to path

    \param fname save path
    """
    mnv, mxv = image.min(), image.max()
    if mxv <= 1.0:
        img = map_image_to_range(image, mnv=0.0, mxv=255.0)
    elif mxv >= 255.0:
        img = map_image_to_range(image, mnv=0.0, mxv=255.0)
    else:
        img = image.copy()
    img = img.astype("uint8")
    im = Image.fromarray(img)
    im.save(fname)
