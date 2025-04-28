"""main.py"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

from computer_vision.src.filter import FilterType, apply_filter, get_filter
from computer_vision.util.images import load_image_as_array, rgb_to_grayscale


@dataclass
class PixelBufferApp:
    _buffer_width: int = 512
    _buffer_height: int = 512

    _buffer: np.ndarray = field(init=False)
    _buffer_history: list[np.ndarray] = field(init=False)
    texture_tag: str = field(init=False)
    last_time: float = field(init=False, default_factory=time.perf_counter)

    data_folder = Path("data")

    def __post_init__(self):
        # initialize raw RGBA buffer
        self._buffer = np.zeros(
            (
                self._buffer_height,
                self._buffer_width,
                4,
            ),
            dtype=np.uint8,
        )
        self._buffer_history = []

        # DearPyGui setup
        dpg.create_context()
        dpg.create_viewport(
            title="Pixel Buffer Viewer", width=900, height=700, resizable=False
        )

        # 1) Create a hidden texture registry for our dynamic texture:
        with dpg.texture_registry(show=False):
            empty_tex = np.zeros(
                (self._buffer_height * self._buffer_width * 4,), dtype=np.float32
            ).tolist()
            self.texture_tag = "dynamic_tex"
            dpg.add_dynamic_texture(
                self._buffer_width,
                self._buffer_height,
                empty_tex,
                tag=self.texture_tag,
            )

        # 2) Now build the actual window UI:
        with dpg.window(label="Pixel Buffer Viewer", tag="Primary Window"):
            dpg.add_text("Simple Raw Pixel Buffer GUI", color=[200, 200, 200])
            dpg.add_separator()

            with dpg.group(horizontal=True):
                # left: the dynamic texture
                dpg.add_image(self.texture_tag)

                # right: stack buttons vertically
                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="Apply Random Noise", callback=self.apply_random_noise
                        )
                        dpg.add_slider_int(
                            default_value=50,
                            min_value=0,
                            max_value=100,
                            width=150,
                            tag="noise_slider",
                        )
                    dpg.add_button(label="Clear Screen", callback=self.clear_screen)
                    dpg.add_button(label="Fill Gradient", callback=self.fill_gradient)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Load Image", callback=self.load_image)

                        choices = [
                            f.name
                            for f in self.data_folder.iterdir()
                            if f.suffix.lower() in (".webp", ".png", ".jpg")
                        ]

                        if not choices:
                            choices = ["(No images found)"]

                        dpg.add_combo(
                            items=choices,
                            default_value=choices[0],
                            fit_width=True,
                            tag="load_image_dropdown",
                        )
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Apply Filter", callback=self.apply_filter)
                        choices = [x.value for x in FilterType]
                        dpg.add_combo(
                            items=choices,
                            default_value=choices[0],
                            fit_width=True,
                            tag="apply_filter_dropdown",
                        )
                    dpg.add_button(label="Undo", callback=self.undo_step)
            dpg.add_separator()

        dpg.set_primary_window("Primary Window", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def fill_gradient(self):
        """Initial gradient fill."""
        for y in range(self._buffer_height):
            for x in range(self._buffer_width):
                self._buffer[y, x] = [x % 256, y % 256, (x + y) % 256, 255]
        self.update_texture()

    def update_texture(self):
        """Normalize to [0,1] floats and push to GPU."""
        flat = (self._buffer.flatten().astype(np.float32) / 255.0).tolist()
        dpg.set_value(self.texture_tag, flat)

    def append_to_history(self) -> None:
        self._buffer_history.append(np.copy(self._buffer))

    def apply_random_noise(self, *args, **kwargs):
        self.append_to_history()

        noise_strength: float = dpg.get_value("noise_slider") / 100.0

        # Generate random noise
        noise: np.ndarray[tuple[int, ...], np.dtype[np.floating[np._32Bit]]] = (
            np.random.randint(
                0, 256, (self._buffer_height, self._buffer_width, 3), dtype=np.uint8
            ).astype(np.float32)
        )

        # Current image (only RGB channels) as float
        current = self._buffer[:, :, :3].astype(np.float32)

        # Blend (linear interpolation)
        blended = (1.0 - noise_strength) * current + noise_strength * noise

        # Clip, round, and convert back to uint8
        self._buffer[:, :, :3] = np.clip(blended, 0, 255).astype(np.uint8)

        # Ensure alpha stays at 255
        self._buffer[:, :, 3] = 255

        self.update_texture()

    def _clear_buffer(self) -> None:
        self._buffer[:, :, :] = 0
        self._buffer[:, :, 3] = 255

    def clear_screen(self, *args, **kwargs):
        self.append_to_history()
        self._clear_buffer()
        self.update_texture()

    def load_image(self, *args, **kwargs):
        self.append_to_history()
        self._clear_buffer()

        image_fp = self.data_folder.joinpath(dpg.get_value("load_image_dropdown"))
        grayscale_image = rgb_to_grayscale(load_image_as_array(image_fp))

        assert len(grayscale_image.shape) == 2
        y_size, x_size = grayscale_image.shape
        y_slice = slice(0, min(self._buffer_height, y_size))
        x_slice = slice(0, min(self._buffer_width, x_size))

        self._buffer[y_slice, x_slice, :3] = grayscale_image[
            y_slice, x_slice, np.newaxis
        ]
        self.update_texture()

    def apply_filter(self, *args, **kwargs) -> None:
        self.append_to_history()
        filter_ = get_filter(FilterType(dpg.get_value("apply_filter_dropdown")))
        filtered_image = apply_filter(
            self._buffer[:, :, 0], filter_=filter_, pad_same_size=True
        )
        self._buffer[:, :, :3] = filtered_image[:, :, np.newaxis]
        self.update_texture()

    def undo_step(self, *args, **kwargs) -> None:
        if len(self._buffer_history) == 0:
            print("Undo buffer empty!")
            return
        self._buffer = np.copy(self._buffer_history.pop())
        self.update_texture()

    def run(self):
        self.fill_gradient()
        dpg.start_dearpygui()
        dpg.destroy_context()


def main():
    app = PixelBufferApp()
    app.run()


if __name__ == "__main__":
    main()
