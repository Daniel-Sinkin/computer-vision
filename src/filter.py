from enum import StrEnum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import DTypeLike

def apply_filter(
    image: np.ndarray, filter: np.ndarray, flip_filter: bool = True
) -> np.ndarray:
    if flip_filter:
        _filter = np.flipud(np.fliplr(filter))
    else:
        _filter = filter

    i_h, i_w = image.shape
    f_h, f_w = filter.shape

    r_h, r_w = i_h - f_h + 1, i_w - f_w + 1

    result = np.zeros((r_h, r_w), dtype=np.float32)

    for y in range(r_h):
        for x in range(r_w):
            region = image[y : y + f_h, x : x + f_w]
            result[y, x] = np.sum(region * _filter)
    return result
