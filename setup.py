"""danielsinkin97@gmail.com"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="computer_vision",
    version="0.1.0",
    author="Daniel Sinkin",
    author_email="danielsinkin97@gmail.com",
    description="A computer vision toolkit.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(
        exclude=["tests", "examples", "exercises", "data", "images"]
    ),
    include_package_data=True,
    install_requires=[
        "numpy>=2.2.4",
        "Pillow>=11.1.0",
        "matplotlib>=3.10.1",
        "requests>=2.32.3",
    ],
    extras_require={
        "notebook": [
            "ipykernel>=6.29.5",
            "ipywidgets>=8.1.5",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
