#!/usr/bin/env python3
"""danielsinkin97@gmail.com"""

import matplotlib.pyplot as plt
import numpy as np

from computer_vision.src.autocorrelation import (
    get_autocorrelation_error_matrix,
    get_autocorrelation_matrix,
)
from computer_vision.src.constants import FolderPath
from computer_vision.src.filter import FilterType, apply_filter, get_filter
from computer_vision.util.images import load_image_as_array, rgb_to_grayscale

image = rgb_to_grayscale(
    load_image_as_array(
        "/Users/danielsinkin/GitHub_private/computer-vision/data/hummingbird.png"
    )
)


def example_autocorrelation_detection(show: bool = True) -> None:
    """Applies autocorrelation in the combined form to the entire image."""
    filter_ = np.array([-2, -1, 0, 1, 2], np.float32)
    ac_matrix = get_autocorrelation_matrix(
        image=image,
        filter_hori=filter_,
        filter_vert=filter_.reshape(-1, 1),
        filter_w=get_filter(FilterType.GAUSS_5X5),
    )
    i_h, i_w = image.shape
    assert ac_matrix.shape == (i_h, i_w, 2, 2)

    delta_u = np.array([10, -5], dtype=np.float32)
    error_matrix = get_autocorrelation_error_matrix(
        ac_matrix=ac_matrix, delta_u=delta_u
    )

    interest_highlighting = apply_filter(
        error_matrix, get_filter(FilterType.GAUSS_15X15), pad_same_size=True
    )
    interest_highlighting = (
        (interest_highlighting - interest_highlighting.min())
        / (interest_highlighting.max() - interest_highlighting.min())
        * 255
    )

    d = 0.2
    for i, d in enumerate(np.linspace(0.0, 1.0, 5)):
        plt.figure(figsize=(6, 9))
        plt.imshow(
            d * image + (1 - d) * interest_highlighting,
            cmap="magma",
            interpolation="nearest",
        )
        plt.axis("off")
        plt.savefig(
            FolderPath.Images.joinpath(f"example_autocorrelation_detection_{i}.png"),
            dpi=300,
        )
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    example_autocorrelation_detection(show=True)
