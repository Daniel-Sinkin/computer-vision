"""danielsinkin97@gmail.com"""

from enum import StrEnum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import DTypeLike


class FilterName(StrEnum):
    SOBEL_X = "sobel_x"
    SOBEL_Y = "sobel_y"
    LAPLACIAN = "laplacian"
    GAUSS_3x3 = "gaussian_3x3"
    GAUSS_5x5 = "gaussian_5x5"
    GAUSS_7x7 = "gaussian_7x7"
    GAUSS_15x15 = "gaussian_15x15"


def get_filter(
    filter_name: FilterName = FilterName.GAUSS_5x5, dtype: DTypeLike = np.float32
) -> np.ndarray:
    match filter_name:
        case FilterName.SOBEL_X:
            return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype)
        case FilterName.SOBEL_Y:
            return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype)
        case FilterName.LAPLACIAN:
            return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=dtype)
        case FilterName.GAUSS_3x3:
            return (1 / 16.0) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=dtype)
        case FilterName.GAUSS_5x5:
            return (1 / 273.0) * np.array(
                [
                    [1, 4, 7, 4, 1],
                    [4, 16, 26, 16, 4],
                    [7, 26, 41, 26, 7],
                    [4, 16, 26, 16, 4],
                    [1, 4, 7, 4, 1],
                ],
                dtype=dtype,
            )
        case FilterName.GAUSS_7x7:
            return (1 / 1003.0) * np.array(
                [
                    [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                    [0.0, 3.0, 13.0, 22.0, 13.0, 3.0, 0.0],
                    [1.0, 13.0, 59.0, 97.0, 59.0, 13.0, 1.0],
                    [2.0, 22.0, 97.0, 159.0, 97.0, 22.0, 2.0],
                    [1.0, 13.0, 59.0, 97.0, 59.0, 13.0, 1.0],
                    [0.0, 3.0, 13.0, 22.0, 13.0, 3.0, 0.0],
                    [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                ],
                dtype=dtype,
            )
        case FilterName.GAUSS_15x15:

            def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
                ax = np.linspace(-(size // 2), size // 2, size)
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
                return kernel / np.sum(kernel)

            return gaussian_kernel(15, sigma=3.0).astype(dtype)
        case _:
            raise ValueError(f"{filter_name=} is not supported!")


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
