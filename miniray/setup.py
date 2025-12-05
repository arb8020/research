"""Setup for MiniRay - Simple distributed computing.

Install with:
    pip install -e .

Or from parent directory:
    pip install -e miniray/
"""

import os

from setuptools import find_packages, setup

setup(
    name="miniray",
    version="0.1.0",
    author="Research",
    description="Simple distributed computing (Heinrich + TCP)",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # No dependencies! Just Python stdlib
        # Optional: PyTorch for NCCL features (but not required)
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "mypy",
        ],
        "nccl": [
            "torch>=2.0.0",  # For multi-node training
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: System :: Distributed Computing",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "miniray-server=miniray.worker_server:main",
        ],
    },
)
