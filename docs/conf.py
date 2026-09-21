"""Build the static website from the repository's existing Markdown."""

import tomllib
from pathlib import Path

metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
project = "Dohnuts"
release = metadata["project"]["version"]
language = "en"

extensions = ["myst_parser"]
root_doc = "index"
include_patterns = [
    "index.md",
    "MODEL_CARD.md",
    "docs/*.md",
    "docs/figures/README.md",
    "results/README.md",
]
exclude_patterns = ["docs/naming-and-writing.md", "docs/versioning.md"]
myst_enable_extensions = ["html_image"]
myst_heading_anchors = 4

html_theme = "furo"
html_title = project
html_logo = "../assets/dohnuts-logo.png"
html_favicon = html_logo
html_static_path = ["_static"]
html_css_files = ["brand.css"]
html_show_sourcelink = False
html_copy_source = False
html_domain_indices = False
html_use_index = False
html_theme_options = {
    "sidebar_hide_name": False,
    "source_repository": "https://github.com/PsiACE/dohnuts/",
    "source_branch": "main",
    "source_directory": "",
    "light_css_variables": {
        "color-brand-primary": "#A52A60",
        "color-brand-content": "#A52A60",
        "color-brand-visited": "#A52A60",
        "color-foreground-primary": "#38241F",
        "color-foreground-secondary": "#66544D",
        "color-foreground-muted": "#66544D",
        "color-background-primary": "#FFF9F4",
        "color-background-secondary": "#F8F0EA",
        "color-background-hover": "#FBE8EF",
        "color-background-border": "#E5D8CF",
        "color-highlight-on-target": "#FBE8EF",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FF9BC0",
        "color-brand-content": "#FF9BC0",
        "color-brand-visited": "#FF9BC0",
        "color-foreground-primary": "#FFF9F4",
        "color-foreground-secondary": "#D2BFB5",
        "color-foreground-muted": "#C7B2A8",
        "color-background-primary": "#241B18",
        "color-background-secondary": "#302420",
        "color-background-hover": "#47302F",
        "color-background-border": "#59453D",
        "color-highlight-on-target": "#47302F",
    },
}
