"""danielsinkin97@gmail.com"""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def load_image_as_array(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)
    with Image.open(path) as img:
        return np.array(img)
