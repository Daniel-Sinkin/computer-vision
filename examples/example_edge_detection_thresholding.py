"""danielsinkin97@gmail.com"""

import numpy as np

from computer_vision.src.filter import FilterType, apply_filter, get_filter
from computer_vision.util.images import (
    load_image_as_array,
    plot_grayscale,
    rgb_to_grayscale,
)


def example_edge_detection_thresholding() -> None:
    """Shows some example edge detection done through (naive) thresholding."""
    image = rgb_to_grayscale(
        load_image_as_array(
            "/Users/danielsinkin/GitHub_private/computer-vision/data/butterfly.webp"
        )
    )
    image = apply_filter(image, get_filter(FilterType.GAUSS_5X5), pad_same_size=True)
    plot_grayscale(np.where(image > 210, 1, 0), filename="edge_detection_thresholding")
    image_sobel_x = apply_filter(
        image, get_filter(FilterType.SOBEL_X), pad_same_size=True
    )
    image_sobel_y = apply_filter(
        image, get_filter(FilterType.SOBEL_Y), pad_same_size=True
    )
    plot_grayscale(
        np.where(image_sobel_x > 5, 0.0, 1.0),
        filename="edge_detection_thresholding_sobel_x",
    )
    plot_grayscale(
        np.where(image_sobel_y > 5, 0.0, 1.0),
        filename="edge_detection_thresholding_sobel_y",
    )


if __name__ == "__main__":
    example_edge_detection_thresholding()
