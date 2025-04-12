"""danielsinkin97@gmail.com"""

import matplotlib.pyplot as plt
import numpy as np

from computer_vision.src.constants import FolderPath
from computer_vision.src.filter import FilterType, apply_filter, get_filter
from computer_vision.util.images import load_image_as_array, rgb_to_grayscale

loaded_image = rgb_to_grayscale(
    load_image_as_array(
        "/Users/danielsinkin/GitHub_private/computer-vision/data/hummingbird.png"
    )
)


def example_autocorrelation_detection(image: np.ndarray) -> None:
    """Applies autocorrelation in the combined form to the entire image."""
    i_h, i_w = image.shape

    filter_ = np.array([-2, -1, 0, 1, 2], np.float32)
    i_x = apply_filter(image, filter_, pad_same_size=True)

    vertical_filter = filter_.reshape(-1, 1)
    i_y = apply_filter(image, vertical_filter, pad_same_size=True)

    w = get_filter(FilterType.GAUSS_5X5)

    i_xx = i_x * i_x
    i_xy = i_x * i_y
    i_yy = i_y * i_y

    a11 = apply_filter(i_xx, w, pad_same_size=True)
    a12 = apply_filter(i_xy, w, pad_same_size=True)
    a22 = apply_filter(i_yy, w, pad_same_size=True)

    a_matrix = np.stack(
        [np.stack([a11, a12], axis=-1), np.stack([a12, a22], axis=-1)], axis=-2
    )
    assert a_matrix.shape == (i_h, i_w, 2, 2)  # 2x2 matrix for every pixel

    delta_u = np.array([10, -5], dtype=np.float32)
    autocorrelation_matrix = np.einsum("i, ...ij, j -> ...", delta_u, a_matrix, delta_u)

    interest_highlighting = apply_filter(
        autocorrelation_matrix, get_filter(FilterType.GAUSS_15X15), pad_same_size=True
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
        plt.show()


if __name__ == "__main__":
    example_autocorrelation_detection(image=loaded_image)
