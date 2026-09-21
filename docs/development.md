# Development

Use Python 3.12 and [PDM](https://pdm-project.org/en/latest/#installation) 2.29.2
or a newer 2.x release. The runtime targets the Linux/ROCm environment documented
in the [inference guide](inference.md). The quality and documentation groups can
be installed without model dependencies.

```bash
pdm use 3.12
make install
make check
make test
make build
```

`make install` installs the runtime, training and agent dependencies, and quality
tools. For editing documentation only, use `make docs-install`.
Run `make help` to see the available targets.

## Dependencies

`pyproject.toml` owns the package metadata, dependencies, tool settings, and PDM
commands. `pdm.lock` records the resolution for Python 3.12, including the pinned
Bub Git revision and the ROCm package artifacts. The published experiment's
environment snapshot remains historical evidence, not an installation input.

| Selection | Purpose |
| --- | --- |
| Default dependencies | Model runtime |
| `train` extra | Training, evaluation, and the unit tests' data dependencies |
| `agent` extra | The pinned Bub SDK integration |
| `quality` group | Ruff, ty, prek, and pytest |
| `docs` group | Sphinx, Furo, and MyST |
| `plot` group | Rendering saved evaluation results |

After changing dependencies, run `make lock`, review the lock diff, and install
the relevant groups with `pdm install --check`. CI refuses a missing or stale
lock rather than resolving different versions. Avoid replacing a trained run's
environment while it is active.

Keep `use_uv = false` in `pdm.toml`: PDM's experimental uv mode does not support
the package-to-index bindings used for ROCm. Other packages resolve from PyPI.

## Checks and builds

`make check` verifies the lock, repository hygiene, Ruff lint and formatting,
and ty checks for `src` against the installed dependencies. `make lint` runs the
checks that only need the `quality` group. `pdm run prek install` optionally
installs the same hygiene hooks locally; installing hooks is not required to run
the checks.

`make test` runs pytest, including the pinned RLCD loss and
gradient values, prefix-cache updates and gradients, calibration metrics, and
data-split isolation. It does not load
model weights, download data, or prove GPU training/inference acceptance. CI runs
these tests with Hugging Face access disabled. GPU acceptance remains a separate
step for changes to execution or training.

Run an individual file or case with `pdm run pytest tests/test_rlcd.py` or
`pdm run pytest -k gradients`. Test discovery is limited to `tests/` in
`pyproject.toml`.

The test job installs the same locked ROCm wheels, checks types, and runs the tests
on CPU. It removes unused Android and .NET SDKs from its temporary Ubuntu runner to make
room for Torch, and skips the large dependency cache. Quality and website jobs
do not install the model runtime.

`make build` uses `pdm-backend` to produce a wheel and source distribution in
`dist/`. Local model weights, datasets, caches, and experiment runs are excluded.
The CI build does not publish packages. Website validation and deployment use
the separate [website workflow](website.md).

The Make targets and separated quality/test jobs adapt the relevant parts of
[Bub's Makefile](https://github.com/bubbuild/bub/blob/main/Makefile) and
[CI](https://github.com/bubbuild/bub/blob/main/.github/workflows/main.yml).
