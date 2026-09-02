import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "penumbra"
author = f"{project} contributors"
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx_copybutton", "sphinx_design"]
myst_enable_extensions = ["colon_fence", "deflist", "substitution"]
myst_substitutions = {"project": project}
html_theme = "furo"
exclude_patterns = ["_build", "guides/model_stacking.md"]
autodoc_mock_imports = [
    "arviz", "astropy", "corner", "exotic_ld", "jax", "jaxopt", "jaxoplanet",
    "matplotlib", "numpyro", "numpyro_ext", "pandas", "scipy", "tinygp",
]
