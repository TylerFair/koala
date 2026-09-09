project = "Koala"
author = "Tyler Fairnington and Koala contributors"
copyright = "2026, Tyler Fairnington. BSD 3-Clause License"

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
html_logo = "_static/koala_logo.png"
html_favicon = "_static/koala_favicon.png"
html_css_files = ["custom.css"]
html_theme_options = {
    "sidebar_hide_name": True,
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
