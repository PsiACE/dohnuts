# Build the website

The static website uses Sphinx, Furo, and MyST. Sphinx builds the pages and
search index, Furo supplies navigation and responsive layouts, and MyST reads
the existing Markdown. The site uses the README, model card, guides, and saved
figures directly; do not maintain separate copies of their content.

From the repository root, with [PDM](https://pdm-project.org/en/latest/#installation)
2.29.2 or a newer 2.x release and Python 3.12:

```bash
pdm use 3.12
make docs-install
make docs-preview
```

Open `http://localhost:8000`. The generated files are in `build/site`.
`make docs-install` selects only the `docs` group from `pdm.lock`, without
installing the project or its model dependencies. The build does not import Dohnuts,
execute the examples, load models, train, or download datasets.

Keep navigation in the existing MyST `toctree` directives. Use `docs/conf.py`
for theme settings and `docs/_static/brand.css` for small styling changes.
Keep the existing logo and published figures as the visual sources.

Use `make docs` for a build without the preview server. Dependency versions and
commands live in `pyproject.toml`; see [development](development.md) for checks
and lock updates.

## GitHub Pages

The website workflow builds pull requests without publishing. Pushes to `main`
and manual runs on `main` build and deploy the static output to GitHub Pages.
Set the repository's Pages source to **GitHub Actions** before the first deploy.
The deployment uses the repository's `github-pages` environment and respects any
configured protection rules. It requires no custom server, external hosting
account, or repository secret.
