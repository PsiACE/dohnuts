# Build the website

The static website uses Sphinx, Furo, and MyST. Sphinx builds the pages and
search index, Furo supplies navigation and responsive layouts, and MyST reads
the existing Markdown. The site uses the README, model card, guides, and saved
figures directly; do not maintain separate copies of their content.

From the repository root, with Python 3.12 or newer:

```bash
python -m venv .cache/site-venv
.cache/site-venv/bin/python -m pip install -r docs/requirements.txt
.cache/site-venv/bin/python -m sphinx -b html -W --keep-going -d build/doctrees -c docs . build/site
.cache/site-venv/bin/python -m http.server 8000 --directory build/site
```

Open `http://localhost:8000`. The generated files are in `build/site`.
The build installs only documentation dependencies. It does not import Dohnuts,
execute the examples, load models, train, or download datasets.

Keep navigation in the existing MyST `toctree` directives. Use `docs/conf.py`
for theme settings and `docs/_static/brand.css` for small styling changes.
Keep the existing logo and published figures as the visual sources.

## GitHub Pages

The website workflow builds pull requests without publishing. Pushes to `main`
and manual runs on `main` build and deploy the static output to GitHub Pages.
Set the repository's Pages source to **GitHub Actions** before the first deploy.
The deployment uses the repository's `github-pages` environment and respects any
configured protection rules. It requires no custom server, external hosting
account, or repository secret.
