#!/usr/bin/env python3
"""
Setup script for Maxwell-Boltzmann Cognitive Load Model
"""

from setuptools import setup, find_packages

setup(
    name="mb_cognitive_load_model",
    version="1.0.0",
    author="Wes Bailey",
    description="Maxwell-Boltzmann cognitive performance model implementation",
    url="https://github.com/yourusername/mb_cognitive_load_model",
    packages=find_packages(where="code"),
    package_dir={"": "code"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "matplotlib>=3.3.0",
        "scipy>=1.6.0",
        "scikit-learn>=0.24.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
