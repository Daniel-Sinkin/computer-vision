"""danielsinkin97@gmail.com"""

import numpy as np


def inverse_warp(
    image: np.ndarray, trafo: np.ndarray, output_shape: tuple[int, int]
) -> np.ndarray:
    """Applies trafo to the image using inverse warp, returning the resulting image."""
    out_h, out_w = output_shape
    output = np.zeros((out_h, out_w), dtype=np.uint8)

    ys, xs = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    ones = np.ones_like(xs)
    coords = np.stack([xs, ys, ones], axis=-1)
    assert coords.shape == (out_h, out_w, 3)

    coords_flat = coords.reshape(-1, 3).T  # (3, N)

    if trafo.shape == (2, 3):
        trafo_3x3 = np.vstack([trafo, [0, 0, 1]])  # convert to 3x3
    else:
        trafo_3x3 = trafo

    t_inv = np.linalg.inv(trafo_3x3)
    source_coords = t_inv @ coords_flat
    source_coords /= source_coords[2]  # normalize if projective

    x_src, y_src = source_coords[:2].reshape(2, out_h, out_w)

    x_src_round = np.round(x_src).astype(int)
    y_src_round = np.round(y_src).astype(int)

    mask = (
        (x_src_round >= 0)
        & (x_src_round < image.shape[1])
        & (y_src_round >= 0)
        & (y_src_round < image.shape[0])
    )

    output[mask] = image[y_src_round[mask], x_src_round[mask]]

    return output
