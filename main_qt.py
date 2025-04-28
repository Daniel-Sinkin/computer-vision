"""main.py"""

import sys

import numpy as np
from PyQt5 import QtGui, QtWidgets


class PixelBufferWidget(QtWidgets.QLabel):
    def __init__(self, width: int = 512, height: int = 512, parent=None):
        super().__init__(parent)
        self.width = width
        self.height = height
        # Raw RGBA buffer
        self.buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.setFixedSize(self.width, self.height)
        self.fill_gradient()

    def fill_gradient(self):
        for y in range(self.height):
            for x in range(self.width):
                self.buffer[y, x] = [x % 256, y % 256, (x + y) % 256, 255]
        self.update_image()

    def apply_random_noise(self):
        self.buffer[:, :, :3] = np.random.randint(
            0, 256, (self.height, self.width, 3), dtype=np.uint8
        )
        self.buffer[:, :, 3] = 255
        self.update_image()

    def clear_screen(self):
        self.buffer[:, :, :] = 0
        self.buffer[:, :, 3] = 255
        self.update_image()

    def update_image(self):
        # Wrap raw buffer in a QImage (Format_RGBA8888)
        img = QtGui.QImage(
            self.buffer.data, self.width, self.height, QtGui.QImage.Format_RGBA8888
        )
        pix = QtGui.QPixmap.fromImage(img)
        self.setPixmap(pix)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel Buffer Viewer")

        # The draw area
        self.buffer_widget = PixelBufferWidget()

        # Buttons
        noise_btn = QtWidgets.QPushButton("Apply Random Noise")
        clear_btn = QtWidgets.QPushButton("Clear Screen")
        noise_btn.clicked.connect(self.buffer_widget.apply_random_noise)
        clear_btn.clicked.connect(self.buffer_widget.clear_screen)

        # Layout
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(noise_btn)
        btn_layout.addWidget(clear_btn)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.buffer_widget)
        main_layout.addLayout(btn_layout)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
