"""danielsinkin97@gmail.com"""

from enum import StrEnum

import numpy as np


class PaddingType(StrEnum):
    """Contains the types of padding that have been implemented so far."""

    ZERO = "zero"
    WRAP = "wrap"
    CLAMP = "clamp"
    MIRROR = "mirror"


def apply_padding(
    image: np.ndarray,
    pad: int | tuple[tuple[int, int], tuple[int, int]],
    padding_type: PaddingType = PaddingType.ZERO,
) -> np.ndarray:
    """Apply the selected type of padding to the image and return this padded version.

    The pad parameter can either be:
      - An integer: pad equally on all sides.
      - A tuple of tuples: ((top, bottom), (left, right)) for specific border sizes.
    """
    if isinstance(pad, int):
        top = pad
        bottom = pad
        left = pad
        right = pad
    else:
        (top, bottom), (left, right) = pad

    i_h, i_w = image.shape
    p_h, p_w = i_h + top + bottom, i_w + left + right

    image_padded = np.zeros((p_h, p_w), dtype=image.dtype)

    image_padded[top : top + i_h, left : left + i_w] = image

    match padding_type:
        case PaddingType.ZERO:
            # For zero padding, nothing additional is needed.
            pass

        case PaddingType.WRAP:
            # Top border
            image_padded[:top, left : left + i_w] = image[-top:, :]
            # Bottom border
            image_padded[top + i_h :, left : left + i_w] = image[:bottom, :]
            # Left border
            image_padded[top : top + i_h, :left] = image[:, -left:]
            # Right border
            image_padded[top : top + i_h, left + i_w :] = image[:, :right]

            # Top-left corner
            image_padded[:top, :left] = image[-top:, -left:]
            # Top-right corner
            image_padded[:top, left + i_w :] = image[-top:, :right]
            # Bottom-left corner
            image_padded[top + i_h :, :left] = image[:bottom, -left:]
            # Bottom-right corner
            image_padded[top + i_h :, left + i_w :] = image[:bottom, :right]

        case PaddingType.CLAMP:
            # Top border: repeat the first row.
            image_padded[:top, left : left + i_w] = image[0, :]
            # Bottom border: repeat the last row.
            image_padded[top + i_h :, left : left + i_w] = image[-1, :]
            # Left border: repeat the first column.
            image_padded[top : top + i_h, :left] = np.tile(
                image[:, 0][:, np.newaxis], (1, left)
            )
            # Right border: repeat the last column.
            image_padded[top : top + i_h, left + i_w :] = np.tile(
                image[:, -1][:, np.newaxis], (1, right)
            )

            # Top-left corner
            image_padded[:top, :left] = image[0, 0]
            # Top-right corner
            image_padded[:top, left + i_w :] = image[0, -1]
            # Bottom-left corner
            image_padded[top + i_h :, :left] = image[-1, 0]
            # Bottom-right corner
            image_padded[top + i_h :, left + i_w :] = image[-1, -1]

        case PaddingType.MIRROR:
            # Top border
            image_padded[:top, left : left + i_w] = image[top:0:-1, :]
            # Bottom border: mirror rows from the bottom.
            image_padded[top + i_h :, left : left + i_w] = image[
                -2 : -2 - bottom : -1, :
            ]
            # Left border
            image_padded[top : top + i_h, :left] = image[:, left:0:-1]
            # Right border
            image_padded[top : top + i_h, left + i_w :] = image[:, -2 : -2 - right : -1]

            # Top-left corner
            image_padded[:top, :left] = image[top:0:-1, left:0:-1]
            # Top-right corner
            image_padded[:top, left + i_w :] = image[top:0:-1, -2 : -2 - right : -1]
            # Bottom-left corner
            image_padded[top + i_h :, :left] = image[-2 : -2 - bottom : -1, left:0:-1]
            # Bottom-right corner
            image_padded[top + i_h :, left + i_w :] = image[
                -2 : -2 - bottom : -1, -2 : -2 - right : -1
            ]

    return image_padded
