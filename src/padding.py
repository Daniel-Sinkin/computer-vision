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
    image: np.ndarray, pad: int, padding_type: PaddingType = PaddingType.ZERO
) -> np.ndarray:
    """Apply the selected type of padding to the image and return this padded version."""
    i_h, i_w = image.shape
    p_h, p_w = i_h + 2 * pad, i_w + 2 * pad

    image_padded = np.zeros((p_h, p_w))
    image_padded[pad:-pad, pad:-pad] = image
    match padding_type:
        case PaddingType.ZERO:
            pass  # Nothing more to do for zero padding
        case PaddingType.WRAP:
            # Center Top
            image_padded[:pad, pad:-pad] = image[-pad:, :]
            # Center Bottom
            image_padded[-pad:, pad:-pad] = image[:pad, :]
            # Center Left
            image_padded[pad:-pad, :pad] = image[:, -pad:]
            # Center Right
            image_padded[pad:-pad, -pad:] = image[:, :pad]

            # Corner Top Left
            image_padded[:pad, :pad] = image[-pad:, -pad:]
            # Corner Top Right
            image_padded[:pad, -pad:] = image[-pad:, :pad]
            # Corner Bottom Left
            image_padded[-pad:, :pad] = image[:pad, -pad:]
            # Corner Bottom Right
            image_padded[-pad:, -pad:] = image[:pad, :pad]
        case PaddingType.CLAMP:
            image_padded[pad:-pad, pad:-pad] = image

            # Center Top
            image_padded[:pad, pad:-pad] = image[0, :]
            # Center Bottom
            image_padded[-pad:, pad:-pad] = image[-1, :]
            # Center Left
            image_padded[pad:-pad, :pad] = np.tile(image[:, 0][:, np.newaxis], (1, pad))
            # Center Right
            image_padded[pad:-pad, -pad:] = np.tile(
                image[:, -1][:, np.newaxis], (1, pad)
            )

            # Corner Top Left
            image_padded[:pad, :pad] = image[0, 0]
            # Corner Top Right
            image_padded[:pad, -pad:] = image[0, -1]
            # Corner Bottom Left
            image_padded[-pad:, :pad] = image[-1, 0]
            # Corner Bottom Right
            image_padded[-pad:, -pad:] = image[-1, -1]
        case PaddingType.MIRROR:
            # Center Top
            image_padded[:pad, pad:-pad] = image[pad:0:-1, :]

            # Center Bottom
            image_padded[-pad:, pad:-pad] = image[-2 : -2 - pad : -1, :]

            # Center Left
            image_padded[pad:-pad, :pad] = image[:, pad:0:-1]

            # Center right
            image_padded[pad:-pad, -pad:] = image[:, -2 : -2 - pad : -1]

            # Corner Top Left
            image_padded[:pad, :pad] = image[pad:0:-1, pad:0:-1]
            # Corner Top Right
            image_padded[:pad, -pad:] = image[pad:0:-1, -2 : -2 - pad : -1]
            # Corner Bottom Left
            image_padded[-pad:, :pad] = image[-2 : -2 - pad : -1, pad:0:-1]
            # Corner Bottom Right
            image_padded[-pad:, -pad:] = image[-2 : -2 - pad : -1, -2 : -2 - pad : -1]

    return image_padded
