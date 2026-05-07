# BERNN Installation

BERNN now supports Python 3.11 and newer.

## Requirements Profiles

Use the new profile files under `requirements/` depending on your goal.

- `requirements/stable.txt`: pinned baseline dependencies (default profile)
- `requirements/dev.txt`: stable profile + test/developer tooling
- `requirements/py311.txt`: Python 3.11 profile
- `requirements/py312.txt`: Python 3.12 profile
- `requirements/py313.txt`: Python 3.13 profile

## Quick Start

Install package extras (recommended):

```bash
pip install "bernn[full]"
```

Or install from requirements profiles:

```bash
pip install -r requirements/stable.txt
pip install -r requirements/dev.txt
```

## Python Version Support

- Python 3.11
- Python 3.12
- Python 3.13

## Common Dependency Notes

### TensorFlow and Typing Extensions

If you see dependency conflicts around `typing-extensions`, use the Python-version extras:

```bash
pip install "bernn[py311-plus]"
pip install "bernn[py312-plus]"
pip install "bernn[py313-plus]"
```

### R / rpy2 Integration

R integration requires a working R runtime and headers (`r-base`, `r-base-dev`) in the environment where dependencies are installed.

If `rpy2` build or import fails:

1. Verify `R --version` works.
2. Ensure system R development packages are installed.
3. Reinstall rpy2 with a system compiler (for example `CC=gcc`).

### ax-platform Conflicts

`ax-platform` can be strict about transitive versions. If conflicts appear, use BERNN extras that exclude AX where possible, or isolate AX usage in a dedicated environment.
