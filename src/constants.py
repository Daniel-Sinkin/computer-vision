"""danielsinkin97@gmail.com"""

from dataclasses import dataclass
from pathlib import Path

RBG_MAX = 255


@dataclass(frozen=True)
class FolderPath:
    """Contains the paths in the library, all relative to the root."""

    Images = Path("images")
