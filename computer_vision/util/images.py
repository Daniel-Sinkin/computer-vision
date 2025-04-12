"""danielsinkin97@gmail.com"""

from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    """
    Takes in an image and converts it to grayscale based on https://en.wikipedia.org/wiki/Rec._601
    """
    rgb = rgb_image.astype(np.float32)
    grayscale = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    return grayscale


def load_image_as_array(path: Path) -> np.ndarray:
    """Loads an image and returns it as a numpy array."""
    path = Path(path)
    with Image.open(path) as img:
        return np.array(img)
