"""danielsinkin97@gmail.com"""

from src.padding import PaddingType, apply_padding
from src.util_image import show_grayscale
from util.image_to_np import load_image_as_array
from util.rbg_to_grayscale import rgb_to_grayscale

image = rgb_to_grayscale(
    load_image_as_array(
        "/Users/danielsinkin/GitHub_private/computer-vision/data/lion_downscaled.jpg"
    )
)


def apply_padding_and_plot(padding_size: int, padding_type: PaddingType):
    """Helper function that pads and plots in one go."""
    filename = f"{padding_type}_padding"
    show_grayscale(
        apply_padding(image=image, pad=padding_size, padding_type=padding_type),
        title=filename,
        filename=filename,
    )


def main() -> None:
    """Displays and saves the different types of padding."""
    apply_padding_and_plot(5, "zero")
    apply_padding_and_plot(25, "one")
    apply_padding_and_plot(10, "wrap")
    apply_padding_and_plot(10, "clamp")


if __name__ == "__main__":
    main()
