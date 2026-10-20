import os

project = "Koala"
author = "Tyler Fairnington and Koala contributors"
copyright = "2026, Tyler Fairnington. BSD 3-Clause License"

# Read the Docs sets these; they give the theme canonical URLs and the
# version switcher context.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
if os.environ.get("READTHEDOCS", "") == "True":
    html_context = {"READTHEDOCS": True}

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "substitution"]
myst_heading_anchors = 3
myst_substitutions = {"project": project}

master_doc = "index"
language = "en"
exclude_patterns = ["_build"]

# Same theme family as the jaxoplanet documentation.
html_theme = "sphinx_book_theme"
html_title = "Koala documentation"
html_logo = "_static/koala_logo.png"
html_favicon = "_static/koala_favicon.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_theme_options = {
    "repository_url": "https://github.com/TylerFair/koala",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": False,
    "use_fullscreen_button": False,
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "logo": {"alt_text": "Koala", "link": "index"},
}
