# PIXI_GITHUB.md
Use this when the package is not published on PyPI (or you need a specific branch/commit) but you still want Pixi to manage it as a locked, reproducible dependency.

## TL;DR

```bash
pixi add --pypi "pkg_name @ git+https://github.com/ORG/REPO.git@<ref>"
````

Example:

```bash
pixi add --pypi "my_pkg @ git+https://github.com/myorg/my_pkg.git@main"
```

---

## Adding a PyPI Dependency from GitHub in Pixi

Pixi supports installing Python packages directly from a GitHub repository using standard PEP 508 syntax (same as `pip`).

Where `<ref>` can be:

* A branch: `@main`
* A tag: `@v1.2.3`
* A commit SHA: `@a1b2c3d`

---

### Manual Manifest Edit

You can also add directly in `pyproject.toml`:

```toml
[project]
dependencies = [
    "my_pkg @ git+https://github.com/myorg/my_pkg.git@main",
]
```

---

# FAQ

## failed to solve
Error:   × failed to solve the pypi requirements of environment 'default' for platform 'linux-64'
  ╰─▶ unexpected panic during PyPI resolution: Failed to do lookahead resolution: Failed to download and build
  `repo @ git+https://url.git@main`

try to see if the branch is master, not main. after user confirmation, try again with master/main


# Reproducibility Recommendation

Prefer:

@vX.Y.Z

or commit SHA instead of a floating branch like `@main`.
