"""danielsinkin97@gmail.com"""

import numpy as np


def rgb_to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    """
    Takes in a rbg image and converts it to grayscale based on https://en.wikipedia.org/wiki/Rec._601
    """
    rgb = rgb_image.astype(np.float32)
    grayscale = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    return grayscale
