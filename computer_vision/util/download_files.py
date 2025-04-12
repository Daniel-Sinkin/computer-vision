"""danielsinkin97@gmail.com"""

from pathlib import Path

import requests

LION_URL = (
    "https://images.pexels.com/photos/18199766/pexels-photo-18199766/"
    "free-photo-of-lion-sitting-on-a-grass.jpeg"
    "?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
)
destination = Path(__file__).parent.joinpath("../data/lion.jpg").resolve()


def download_files() -> None:
    """Download the files."""
    response = requests.get(LION_URL, timeout=10.0)
    response.raise_for_status()

    destination.parent.mkdir(parents=True, exist_ok=True)

    with open(destination, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    download_files()
