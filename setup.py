"""Discover Koala packages; metadata is in pyproject.toml."""

from setuptools import find_packages, setup


setup(packages=find_packages(include=["koala", "koala.*", "models", "models.*"]))
