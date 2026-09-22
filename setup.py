"""T2Music Phase-A research kernel (editable install)."""

from setuptools import setup, find_packages

setup(
    name="t2music",
    version="0.1.0-phase-a",
    description="T2Music Phase-A: text-to-music generation research kernel",
    author="t2music-team",
    packages=find_packages(where=".", exclude=["tests*", "docs*"]),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[],
    extras_require={
        "dev": ["pytest", "black", "isort", "ruff"],
        "eval": ["frechet-audio-distance==0.3.2", "laion-clap==1.1.6"],
    },
)
