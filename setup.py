from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="humanml3d",
    version="0.0.0",
    description="HumanML3D preprocessing and visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chanhyuk Jung",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=["numpy", "torch>=2.0.0", "einops"],
    extras_require={
        "test": ["pytest"],
        "dev": ["black", "flake8", "bumpver"],
    },
)
