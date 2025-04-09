"""danielsinkin97@gmail.com"""

import numpy as np

from src.filter import FilterName, apply_filter, get_filter
from src.util_image import show_grayscale
from util.image_to_np import load_image_as_array
from util.rbg_to_grayscale import rgb_to_grayscale

image_loaded = rgb_to_grayscale(
    load_image_as_array(
        "/Users/danielsinkin/GitHub_private/computer-vision/data/lion_downscaled.jpg"
    )
)


def get_my_image() -> np.ndarray:
    return np.array(
        [
            [45, 60, 98, 127, 132, 133, 137, 133],
            [46, 65, 98, 123, 126, 128, 131, 133],
            [47, 65, 96, 115, 119, 123, 135, 137],
            [47, 63, 91, 107, 113, 122, 138, 134],
            [50, 59, 80, 97, 110, 123, 133, 134],
            [49, 53, 68, 83, 97, 113, 128, 133],
            [50, 50, 58, 70, 84, 102, 116, 126],
            [50, 50, 52, 58, 69, 86, 101, 120],
        ],
        dtype=np.float32,
    )


def get_my_filter() -> np.ndarray:
    return np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]])


def filter_and_plot(filter_name):
    show_grayscale(
        apply_filter(image_loaded, get_filter(filter_name)),
        title=filter_name,
        filename=filter_name,
    )


def main() -> None:
    image_conv = apply_filter(image=get_my_image(), filter=get_my_filter())

    show_grayscale(get_my_image())
    show_grayscale(image_conv)

    show_grayscale(image_loaded, title="original")
    filter_and_plot(filter_name=FilterName.SOBEL_X)
    filter_and_plot(filter_name=FilterName.SOBEL_Y)

    filter_and_plot(filter_name=FilterName.LAPLACIAN)

    smoothed_image = apply_filter(image_loaded, get_filter(FilterName.GAUSS_5x5))
    show_grayscale(
        apply_filter(smoothed_image, get_filter(FilterName.LAPLACIAN)),
        title="Smoothed (5x5) then Laplace",
    )
    show_grayscale(
        apply_filter(smoothed_image, get_filter(FilterName.SOBEL_X)),
        title="Smoothed (5x5) then Sobel_x",
    )
    show_grayscale(
        apply_filter(smoothed_image, get_filter(FilterName.SOBEL_Y)),
        title="Smoothed (5x5) then Sobel_y",
    )

    for gaussian in [
        FilterName.GAUSS_3x3,
        FilterName.GAUSS_5x5,
        FilterName.GAUSS_7x7,
        FilterName.GAUSS_15x15,
    ]:
        filter_and_plot(filter_name=gaussian)


if __name__ == "__main__":
    main()
