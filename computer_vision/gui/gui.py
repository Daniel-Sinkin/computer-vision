"""danielsinkin97@gmail.com"""

import logging
import time
from pathlib import Path
from typing import Literal

import dearpygui.dearpygui as dpg
import numpy as np

from computer_vision.src.filter import FilterType, apply_filter, get_filter
from computer_vision.util.images import load_image_as_array, rgb_to_grayscale


class DPGHandler(logging.Handler):
    """Redirects Python logging records to a DearPyGui multiline text box."""

    def __init__(self, textbox_tag: str) -> None:
        super().__init__()
        self.textbox_tag = textbox_tag

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        try:
            existing = dpg.get_value(self.textbox_tag) or ""
            dpg.set_value(self.textbox_tag, f"{msg}\n{existing}")  # prepend newest
        except Exception:
            # UI not ready or another problem—quietly ignore
            pass


class PixelBufferApp:
    """Interactive RGBA pixel-buffer playground with DearPyGui."""

    def __init__(self, buffer_width: int = 512, buffer_height: int = 512) -> None:
        self._buffer_width = buffer_width
        self._buffer_height = buffer_height

        # The image information
        self._buffer: np.ndarray = self.get_empty_buffer()
        # What we drew, will be drawn on top of the core buffer, maybe replaced by layering later
        self._buffer_draw = self.get_empty_buffer()
        self._buffer_history: list[np.ndarray] = []

        self.texture_tag: str = ""
        self.last_time: float = time.perf_counter()
        self.data_folder = Path("data")

        dpg.create_context()
        dpg.create_viewport(
            title="Pixel Buffer Viewer", width=900, height=700, resizable=False
        )

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

        with dpg.window(label="Pixel Buffer Viewer", tag="Primary Window"):
            with dpg.group(horizontal=True):
                dpg.add_image(self.texture_tag)

                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="Apply Random Noise",
                            callback=self.apply_random_noise,
                        )
                        dpg.add_slider_int(
                            default_value=50,
                            min_value=0,
                            max_value=100,
                            width=150,
                            tag="noise_slider",
                        )
                    dpg.add_button(label="Clear Screen", callback=self.clear_screen)
                    dpg.add_button(
                        label="Clear Draw Buffer", callback=self.clear_draw_buffer
                    )

                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Load Image", callback=self.load_image)
                        img_choices = [
                            f.name
                            for f in self.data_folder.iterdir()
                            if f.suffix.lower() in (".webp", ".png", ".jpg", ".jpeg")
                        ]
                        if not img_choices:
                            img_choices = ["(No images found)"]
                        dpg.add_combo(
                            items=img_choices,
                            default_value=img_choices[0],
                            fit_width=True,
                            tag="load_image_dropdown",
                        )

                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Apply Filter", callback=self.apply_filter)
                        filter_choices = [ft.value for ft in FilterType]
                        dpg.add_combo(
                            items=filter_choices,
                            default_value=filter_choices[0],
                            fit_width=True,
                            tag="apply_filter_dropdown",
                        )

                    dpg.add_button(label="Undo", callback=self.undo_step)

            dpg.add_separator()
            dpg.add_input_text(
                label="Logs",
                tag="log_box",
                multiline=True,
                readonly=True,
                height=150,
                width=-1,
            )

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(callback=self.on_mouse_click)
            dpg.add_mouse_drag_handler(callback=self.on_mouse_drag, threshold=0)
            dpg.add_mouse_wheel_handler(callback=self.on_mouse_wheel)

        self._configure_logging()

        dpg.set_primary_window("Primary Window", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        self.clear_screen()

    def _configure_logging(self) -> None:
        self.logger = logging.getLogger("PixelBufferApp")
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )

        gui_handler = DPGHandler("log_box")
        gui_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(gui_handler)
        self.logger.addHandler(console_handler)

    def get_empty_buffer(self) -> np.ndarray:
        return np.zeros((self._buffer_height, self._buffer_width, 4), dtype=np.uint8)

    def _clear_buffer(self) -> None:
        self._buffer[:, :, :] = 0
        self._buffer[:, :, 3] = 255

    def _push_to_gpu(self) -> None:
        """Normalize to [0,1] floats and copy into the DearPyGui dynamic texture."""
        base = np.copy(self._buffer)
        draw = self._buffer_draw

        mask = np.any(draw != 0, axis=-1)
        base[mask] = draw[mask]

        flat = base.flatten().astype(np.float32) / 255.0
        dpg.set_value(self.texture_tag, flat.tolist())

    def _append_to_history(self) -> None:
        self._buffer_history.append(np.copy(self._buffer))

    def clear_draw_buffer(self, *_, **__) -> None:
        self._buffer_draw = self.get_empty_buffer()
        self._push_to_gpu()

    def apply_random_noise(self, *_, **__) -> None:
        self._append_to_history()
        strength = dpg.get_value("noise_slider") / 100.0

        noise: np.ndarray[tuple[int, ...], np.dtype[np.floating[np._32Bit]]] = (
            np.random.randint(
                0,
                256,
                (self._buffer_height, self._buffer_width, 3),
                dtype=np.uint8,
            ).astype(np.float32)
        )

        current = self._buffer[:, :, :3].astype(np.float32)
        blended = (1.0 - strength) * current + strength * noise
        self._buffer[:, :, :3] = np.clip(blended, 0, 255).astype(np.uint8)
        self._buffer[:, :, 3] = 255
        self._push_to_gpu()

    def clear_screen(self, *_, **__) -> None:
        self._append_to_history()
        self._clear_buffer()
        self._push_to_gpu()

    def load_image(self, *_, **__) -> None:
        self._append_to_history()
        self._clear_buffer()

        img_name = dpg.get_value("load_image_dropdown")
        img_path = self.data_folder.joinpath(img_name)
        gray = rgb_to_grayscale(load_image_as_array(img_path))

        y_max, x_max = gray.shape
        y_slice = slice(0, min(self._buffer_height, y_max))
        x_slice = slice(0, min(self._buffer_width, x_max))

        self._buffer[y_slice, x_slice, :3] = gray[y_slice, x_slice, np.newaxis]
        self._push_to_gpu()

    def apply_filter(self, *_, **__) -> None:
        self._append_to_history()
        filter_name = dpg.get_value("apply_filter_dropdown")
        filter_kernel = get_filter(FilterType(filter_name))
        filtered = apply_filter(
            self._buffer[:, :, 0], filter_=filter_kernel, pad_same_size=True
        )
        self._buffer[:, :, :3] = filtered[:, :, np.newaxis]
        self._push_to_gpu()

    def undo_step(self, *_, **__) -> None:
        if not self._buffer_history:
            self.logger.warning("Undo buffer empty!")
            return
        self._buffer = np.copy(self._buffer_history.pop())
        self._push_to_gpu()

    def color_pixel(self, ty: int, tx: int, color: Literal["r", "g", "b"]) -> None:
        match color:
            case "r":
                color_idx = 0
            case "g":
                color_idx = 1
            case "b":
                color_idx = 2

        self._buffer_draw[ty - 5 : ty + 5, tx - 5 : tx + 5, [0, 3]] = 255.0
        self._push_to_gpu()

    def on_mouse_click(self, sender, app_data) -> None:
        # Global mouse pos
        mx, my = dpg.get_mouse_pos(local=False)
        # Workaround to the fact that dpg adds 10 pixel paddin
        mx -= 10
        my -= 10

        # Texture position and size
        x0, y0 = dpg.get_item_pos(self.texture_tag)
        w = dpg.get_item_width(self.texture_tag)
        h = dpg.get_item_height(self.texture_tag)

        # Bail if outside
        if mx < x0 or mx > x0 + w or my < y0 or my > y0 + h:
            return

        # Map to texels
        tx = int((mx - x0) / w * self._buffer_width)
        ty = int((my - y0) / h * self._buffer_height)
        tx = max(0, min(self._buffer_width - 1, tx))
        ty = max(0, min(self._buffer_height - 1, ty))
        self.logger.info("Clicked texel at: (%d, %d)", ty, tx)
        self.color_pixel(ty=ty, tx=tx, color="r")

    def on_mouse_drag(self, sender, app_data) -> None:
        pass

    def on_mouse_wheel(self, sender, app_data) -> None:
        pass

    def run(self) -> None:
        dpg.start_dearpygui()
        dpg.destroy_context()
