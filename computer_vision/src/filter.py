"""danielsinkin97@gmail.com"""

from enum import StrEnum

import numpy as np
from numpy._typing import DTypeLike


class FilterType(StrEnum):
    """Contains all implemented linear filters."""

    BOX_5 = "box_5"
    BOX_10 = "box_10"
    BILINEAR = "bilinear"
    SOBEL_X = "sobel_x"
    SOBEL_Y = "sobel_y"
    CORNER = "corner"
    LAPLACIAN = "laplacian"
    GAUSS_3X3 = "gaussian_3x3"
    GAUSS_5X5 = "gaussian_5x5"
    GAUSS_7X7 = "gaussian_7x7"
    GAUSS_15X15 = "gaussian_15x15"


def get_filter(
    filter_name: FilterType = FilterType.GAUSS_5X5, dtype: DTypeLike = np.float32
) -> np.ndarray:
    """Allows selecting filters via their name."""
    match filter_name:
        case FilterType.BOX_5:
            filter_ = (1.0 / 25.0) * np.ones((5, 5))
        case FilterType.BOX_10:
            filter_ = (1.0 / 100.0) * np.ones((10, 10))
        case FilterType.BILINEAR:
            filter_ = (1.0 / 16.0) * np.array(
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=dtype
            )
        case FilterType.CORNER:
            filter_ = (1.0 / 4.0) * np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
        case FilterType.SOBEL_X:
            filter_ = (1.0 / 8.0) * np.array(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype
            )
        case FilterType.SOBEL_Y:
            filter_ = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype)
        case FilterType.LAPLACIAN:
            filter_ = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=dtype)
        case FilterType.GAUSS_3X3:
            filter_ = (1.0 / 16.0) * np.array(
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=dtype
            )
        case FilterType.GAUSS_5X5:
            filter_ = (1.0 / 256.0) * np.array(
                [
                    [1, 4, 6, 4, 1],
                    [4, 16, 24, 16, 4],
                    [6, 24, 36, 24, 6],
                    [4, 16, 24, 16, 4],
                    [1, 4, 6, 4, 1],
                ],
                dtype=dtype,
            )
        case FilterType.GAUSS_7X7:
            filter_ = (1.0 / 1003.0) * np.array(
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
        case FilterType.GAUSS_15X15:

            def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
                ax = np.linspace(-(size // 2), size // 2, size)
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
                return kernel / np.sum(kernel)

            filter_ = gaussian_kernel(15, sigma=3.0).astype(dtype)
        case _:
            raise ValueError(f"{filter_name=} is not supported!")

    return filter_


def apply_filter(
    image: np.ndarray,
    filter_: np.ndarray,
    flip_filter: bool = True,
    pad_same_size: bool = False,
) -> np.ndarray:
    """
    Applies a linear filter (1D or 2D) to the given image, either normally or flipped.
    """
    if filter_.ndim == 1:
        # Assume 1D filters are horizontal by default
        filter_ = filter_.reshape(1, -1)

    if flip_filter:
        _filter = np.flipud(np.fliplr(filter_))
    else:
        _filter = filter_

    f_h, f_w = _filter.shape

    if pad_same_size:
        pad_h, pad_w = f_h // 2, f_w // 2
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    i_h, i_w = image.shape

    r_h, r_w = i_h - f_h + 1, i_w - f_w + 1
    result = np.zeros((r_h, r_w), dtype=np.float32)

    for y in range(r_h):
        for x in range(r_w):
            region = image[y : y + f_h, x : x + f_w]
            result[y, x] = np.sum(region * _filter)

    return result
