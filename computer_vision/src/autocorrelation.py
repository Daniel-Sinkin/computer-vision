"""danielsinkin97@gmail.com"""

import numpy as np

from .filter import apply_filter


def get_autocorrelation_matrix(
    image: np.ndarray,
    filter_hori: np.ndarray,
    filter_vert: np.ndarray,
    filter_w: np.ndarray,
) -> np.ndarray:
    """See equation (4.8) in the book."""
    i_x = apply_filter(image, filter_hori, pad_same_size=True)
    i_y = apply_filter(image, filter_vert, pad_same_size=True)

    i_xx = i_x * i_x
    i_xy = i_x * i_y
    i_yy = i_y * i_y

    a11 = apply_filter(i_xx, filter_w, pad_same_size=True)
    a12 = apply_filter(i_xy, filter_w, pad_same_size=True)
    a22 = apply_filter(i_yy, filter_w, pad_same_size=True)

    return np.stack(
        [np.stack([a11, a12], axis=-1), np.stack([a12, a22], axis=-1)], axis=-2
    )


def get_autocorrelation_error_matrix(
    ac_matrix: np.ndarray, delta_u: np.ndarray
) -> np.ndarray:
    """See equation (4.6) in the book."""
    return np.einsum("i, ...ij, j -> ...", delta_u, ac_matrix, delta_u)
