"""danielsinkin97@gmail.com"""

import sys
import time
from dataclasses import dataclass

import sdl2
import sdl2.ext

# Global state variables
MOUSE_LEFT_DOWN = False

SHIFT_DOWN = False

held_object = None  # Keeps track of the currently dragged object
held_object_offset = None

window_flags = sdl2.SDL_WINDOW_SHOWN

sdl2.ext.init()
window = sdl2.ext.Window("SDL2 Window", size=(800, 600), flags=window_flags)
sdl2.SDL_SetWindowResizable(window.window, sdl2.SDL_FALSE)  # Disable resizing
window.show()

renderer = sdl2.ext.Renderer(window)


def get_mouse_pos() -> tuple[int, int]:
    m_x = sdl2.Sint32()
    m_y = sdl2.Sint32()
    sdl2.SDL_GetMouseState(m_x, m_y)
    return (m_x.value, m_y.value)


@dataclass
class Rectangle:
    """Anchor is top left"""

    x: int
    y: int
    w: int
    h: int
    color: tuple[int, int, int]

    def get_center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def is_hover(self) -> bool:
        m_x, m_y = get_mouse_pos()
        inside_x_range = (m_x >= self.x) and (m_x <= self.x + self.w)
        inside_y_range = (m_y >= self.y) and (m_y <= self.y + self.h)
        return inside_x_range and inside_y_range

    def on_mouse_hold(self) -> None:
        # Removed hover check here so the object continues to follow the mouse
        m_x, m_y = get_mouse_pos()
        self.x = m_x - self.w // 2 - held_object_offset[0]
        self.y = m_y - self.h // 2 - held_object_offset[1]

    def on_click(self) -> None:
        # Additional click behavior can be added here; currently, it is a no-op.
        print("Clicked Button!")

    def to_sdl(self) -> sdl2.SDL_Rect:
        return sdl2.SDL_Rect(
            self.x,
            self.y,
            self.w,
            self.h,
        )

    def render(self) -> None:
        if self.is_hover():
            sdl2.SDL_SetRenderDrawColor(renderer.sdlrenderer, 255, 0, 0, 255)
        else:
            sdl2.SDL_SetRenderDrawColor(renderer.sdlrenderer, *self.color, 255)
        sdl2.SDL_RenderFillRect(renderer.sdlrenderer, self.to_sdl())

    def scale_centered(self, increase_factor: float) -> None:
        dw = int(self.w * increase_factor)
        dh = int(self.h * increase_factor)

        self.x -= dw // 2
        self.y -= dh // 2

        self.w += dw
        self.h += dh


scene = [
    Rectangle(
        x=100 + 100 * i,
        y=300,
        w=50,
        h=50,
        color=(255, 255, 255),
    )
    for i in range(5)
]

running = True

while running:
    events = sdl2.ext.get_events()
    for event in events:
        if event.type == sdl2.SDL_QUIT:
            running = False
            break
        elif event.type == sdl2.SDL_MOUSEBUTTONDOWN:
            if event.button.button == sdl2.SDL_BUTTON_LEFT:
                # Only grab an object if none is already held.
                if not MOUSE_LEFT_DOWN:
                    for i, obj in enumerate(scene):
                        if obj.is_hover():
                            held_object = obj
                            obj_c_x, obj_c_y = obj.get_center()
                            m_x, m_y = get_mouse_pos()
                            held_object_offset = (m_x - obj_c_x, m_y - obj_c_y)
                            obj.on_click()
                            mods = sdl2.SDL_GetModState()
                            shift_held = bool(mods & sdl2.KMOD_SHIFT)
                            if shift_held:
                                # Copy and append
                                scene.append(
                                    Rectangle(
                                        x=obj.x,
                                        y=obj.y,
                                        w=obj.w,
                                        h=obj.h,
                                        color=obj.color,
                                    )
                                )
                            break
                    MOUSE_LEFT_DOWN = True
        elif event.type == sdl2.SDL_MOUSEBUTTONUP:
            if event.button.button == sdl2.SDL_BUTTON_LEFT:
                MOUSE_LEFT_DOWN = False
                held_object = None
        elif event.type == sdl2.SDL_MOUSEWHEEL:
            increase_factor = 0.05 * event.wheel.y
            for obj in scene:
                if obj.is_hover():
                    obj.scale_centered(increase_factor=increase_factor)

    # If the mouse is held and an object is grabbed, update its position.
    if MOUSE_LEFT_DOWN and held_object is not None:
        held_object.on_mouse_hold()

    # print(f"Mouse position: {get_mouse_pos()}")

    renderer.clear(sdl2.ext.Color(0, 0, 0))  # Black background

    for obj in scene:
        obj.render()

    renderer.present()

sdl2.ext.quit()
sys.exit(0)
