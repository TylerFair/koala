"""Build Koala with its bundled Harmonica extension; metadata is in pyproject.toml."""

from pathlib import Path
import runpy

from setuptools import find_packages, setup


vendor = Path("harmonica_modified")
build = runpy.run_path(str(vendor / "setup.py"))
extensions = build["ext_modules"]
for extension in extensions:
    extension.sources = [str(vendor / path) for path in extension.sources]
    extension.include_dirs = [
        str(vendor / path) if not Path(path).is_absolute() else path
        for path in extension.include_dirs
    ]

setup(
    packages=(
        find_packages(include=["koala", "koala.*", "models", "models.*"])
        + find_packages(where=str(vendor), include=["harmonica", "harmonica.*"])
    ),
    package_dir={"harmonica": str(vendor / "harmonica")},
    ext_modules=extensions,
    cmdclass={"build_ext": build["build_ext_with_openmp"]},
)
