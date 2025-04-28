"""danielsinkin97@gmail.com"""
# pylint: disable=unused-argument,missing-docstring,broad-exception-caught

import logging
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import dearpygui.dearpygui as dpg
import numexpr
import numpy as np
from PIL import Image

fp_example = Path("examples")
fp_data = Path("data")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")

MAX_IMG_W, MAX_IMG_H = 800, 600


class GUI:
    def __init__(self) -> None:
        self._setup_logger()
        self.setup_gui()

    def _setup_logger(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    def setup_gui(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Computer Vision", width=1600, height=900)
        dpg.setup_dearpygui()

        font_path = Path(__file__).parent.joinpath(
            "assets", "MonaspaceKrypton-Regular.otf"
        )
        with dpg.font_registry():
            self._font = dpg.add_font(str(font_path), 16)
        with dpg.texture_registry(show=False, tag="__texreg__"):
            pass

        with dpg.window(label="Menu"):
            with dpg.menu(label="Load Image"):
                # populate every *.png|*.jpg|*.webp in data/
                for img_path in fp_data.iterdir():
                    if img_path.suffix.lower() in IMG_EXTS:
                        dpg.add_menu_item(
                            label=img_path.name,
                            callback=self.callback_load_image,
                            user_data=img_path,
                        )

        with dpg.window(label="Run Example Scripts"):
            for func in fp_example.iterdir():
                if func.suffix == ".py" and func.name != "__init__.py":
                    dpg.add_button(
                        label=f"{func.name}",
                        callback=self.callback_btn_invoke_py_script,
                        user_data=func,
                    )

        with dpg.window(label="Syntax Evaluation", pos=(400, 0)):
            dpg.add_input_text(
                label="f(x)",
                tag="syntax_eval_func_input",
                default_value="cos(x)",
                callback=self.callback_ti_syntax_changed,
                on_enter=False,
            )
            dpg.add_input_text(
                label="Value",
                tag="syntax_eval_value_input",
                default_value="0.0",
                callback=self.callback_ti_syntax_changed,
                on_enter=False,
            )
            dpg.add_input_text(
                label="Result",
                tag="syntax_eval_value_output",
                readonly=True,
                default_value="",
            )

        self.evaluate_syntax_change()

        with dpg.handler_registry():
            dpg.add_key_down_handler(callback=self.callback_key_down)

    def run(self) -> None:
        dpg.show_viewport()
        dpg.bind_font(self._font)
        dpg.start_dearpygui()
        self.cleanup()

    def evaluate_syntax_change(self) -> None:
        """Called every time either syntax input box changes."""
        expr = dpg.get_value("syntax_eval_func_input")
        val_text = dpg.get_value("syntax_eval_value_input")

        try:
            x_val = float(val_text)
            result = numexpr.evaluate(expr, local_dict={"x": x_val})
            result_str = str(result.item() if hasattr(result, "item") else result)
            dpg.set_value("syntax_eval_value_output", result_str)
        except Exception:
            dpg.set_value("syntax_eval_value_output", "INVALID")

    def callback_ti_syntax_changed(self, sender, app_data, user_data) -> None:
        self.evaluate_syntax_change()

    def log_subprocess_output(self, stdout, stderr, script_name: str) -> None:
        assert stdout is not None
        assert stderr is not None

        for line in stdout:
            self._logger.info("[%s][stdout] %s", script_name, line.rstrip())

        for line in stderr:
            self._logger.error("[%s][stderr] %s", script_name, line.rstrip())

    def callback_btn_invoke_py_script(self, sender, app_data, user_data: Path) -> None:
        try:
            process = subprocess.Popen(
                [sys.executable, str(user_data)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            threading.Thread(
                target=self.log_subprocess_output,
                args=(process.stdout, process.stderr, user_data.name),
                daemon=True,
            ).start()

            self._logger.info("Launched '%s' successfully.", user_data.name)
        except Exception as e:
            self._logger.error("Failed to launch '%s': %s", user_data.name, e)

    def callback_key_down(self) -> None:
        if dpg.is_key_down(dpg.mvKey_Escape):
            dpg.stop_dearpygui()

    def callback_hover_image(self, sender, app_data, user_data):
        """
        Fires every frame while the mouse is over the image widget.
        user_data = dict(tex_tag=…, img_w=…, img_h=…, view_w=…, view_h=…)
        """
        # widget & mouse positions are in DearPyGui's global pixel space
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        item_x, item_y = dpg.get_item_pos(sender)  # sender *is* the image item
        rel_x = mouse_x - item_x
        rel_y = mouse_y - item_y

        # inside the displayed rectangle?
        vw, vh = user_data["view_w"], user_data["view_h"]
        if 0 <= rel_x < vw and 0 <= rel_y < vh:
            # convert from visible-pixel to original-image texel
            # (top-left 1-to-1 because we clipped, not scaled)
            tex_x = int(rel_x)
            tex_y = int(rel_y)
            self._logger.debug(
                "[%s] hover texel (%d, %d)", user_data["tex_tag"], tex_x, tex_y
            )

    def callback_load_image(self, sender, app_data, user_data: Path) -> None:
        """Menu-item clicked → load image, create texture, show clipped view."""
        try:
            # Load image as RGBA (DearPyGui expects 4 channels)
            img = Image.open(user_data).convert("RGBA")
            img_w, img_h = img.size
            img_data = (np.asarray(img, dtype=np.float32) / 255.0).flatten()
        except Exception as exc:
            self._logger.error("Failed loading image '%s': %s", user_data, exc)
            return

        # Clip (not scale) to MAX_IMG_W × MAX_IMG_H via UV coordinates
        disp_w, disp_h = min(img_w, MAX_IMG_W), min(img_h, MAX_IMG_H)
        uv_max = (disp_w / img_w, disp_h / img_h)

        tex_tag = f"tex::{uuid.uuid4()}"
        dpg.add_static_texture(img_w, img_h, img_data, parent="__texreg__", tag=tex_tag)

        # Window slightly larger than the image so borders/title fit nicely
        win_w, win_h = disp_w + 16, disp_h + 38
        with dpg.window(
            label=f"Image - {user_data.name}",
            width=win_w,
            height=win_h,
            no_resize=True,
        ):
            img_item = dpg.add_image(
                tex_tag,
                width=disp_w,
                height=disp_h,
                uv_min=(0.0, 0.0),
                uv_max=uv_max,
            )

        # -------- item handlers -------------------------------------------------
        with dpg.item_handler_registry(tag=f"{tex_tag}_hover_handlers") as ih_reg:
            dpg.add_item_hover_handler(
                callback=self.callback_hover_image,
                user_data={
                    "tex_tag": tex_tag,
                    "img_w": img_w,
                    "img_h": img_h,
                    "view_w": disp_w,
                    "view_h": disp_h,
                },
            )

        # attach registry to the image widget
        dpg.bind_item_handler_registry(img_item, ih_reg)
        # -----------------------------------------------------------------------

        self._logger.info(
            "Loaded image '%s' (%dx%d → showing %dx%d)",
            user_data.name,
            img_w,
            img_h,
            disp_w,
            disp_h,
        )

    def cleanup(self) -> None:
        dpg.destroy_context()
