project = "Koala"
author = "Koala contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "substitution"]
myst_heading_anchors = 3
myst_substitutions = {"project": project}

html_theme = "furo"
html_title = "Koala: Kool exOplAnet Lightcurve Analysis"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#76538e",
        "color-brand-content": "#76538e",
        "color-admonition-background": "#f7f5f9",
    },
    "dark_css_variables": {
        "color-brand-primary": "#c8a9dc",
        "color-brand-content": "#c8a9dc",
    },
}

exclude_patterns = ["_build"]
