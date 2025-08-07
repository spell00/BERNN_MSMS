# BERNN Installa### Python Version Specific
```bash
pip install bernn[py38]              # Python 3.8 optimized (with modern typing-extensions)
pip install bernn[py39]              # Python 3.9 and earlier optimized
pip install bernn[py310+]            # Python 3.10+ optimized
pip install bernn[py312+]            # Python 3.12+ optimized
pip install bernn[python38-full]     # Python 3.8 full install with modern compatibility
```

### Python 3.8 Specific Options
```bash
pip install bernn[python38-modern]   # Updated typing-extensions for Python 3.8
pip install bernn[python38-full]     # PyTorch + tools (no TensorFlow to avoid conflicts)
pip install bernn[python38-tensorflow] # Full install with TensorFlow (may have typing conflicts)
pip install bernn[python38-ml-minimal] # PyTorch only for Python 3.8
```Options

BERNN now supports multiple installation options to allow users to install only the dependencies they need. The package automatically handles different Python versions (3.8, 3.9, 3.10+) with appropriate dependency versions and includes conflict resolution for common package incompatibilities.

## Quick Start

For most users, we recommend:
```bash
pip install bernn[full]              # Recommended: All features except conflicting packages
```

For a completely safe installation:
```bash
pip install bernn                    # Minimal installation with core features only
```

## Python Version Support

- **Python 3.8-3.9**: Uses pinned versions for maximum compatibility and conflict avoidance
- **Python 3.10-3.11**: Uses newer versions with balanced constraints
- **Python 3.12+**: Uses latest versions optimized for Python 3.12

## Common Conflicts and Solutions

### ax-platform Conflicts
**Problem**: ax-platform requires very old versions of dependencies (aiohttp<=3.6.2, packaging<=20.1, SQLAlchemy<=1.3.13)

**Solutions**:
1. Use `pip install bernn[tools]` instead of `bernn[tools-with-ax]`
2. Install optuna as alternative: `pip install bernn[tools] optuna`
3. Use clean virtual environment for ax-platform

### typing-extensions Conflicts
**Problem**: Multiple packages require different versions of typing-extensions

**Solutions**:
1. Update to compatible version: `pip install bernn[typing]`
2. For TensorFlow 2.13: Use `typing-extensions>=4.5.0,<4.6.0`
3. For modern packages: Use `typing-extensions>=4.9.0`

### Web Framework Conflicts (fastapi, pydantic, spotdl)
**Problem**: spotdl requires newer fastapi/pydantic than some environments have

**Solutions**:
1. Use modern web stack: `pip install bernn[web-dev]`
2. Separate installation: Install spotdl in separate environment
3. Use bernn[web] for basic web features without spotdl compatibility

### IDE Tool Conflicts (spyder, selenium)
**Problem**: spyder requires old PyQt5 versions, selenium needs newer typing-extensions

**Solutions**:
1. Use updated versions: `pip install bernn[ide-tools]`
2. Upgrade spyder to 5.0+: Includes in `bernn[ide-tools]`
3. Constrain selenium version: Use selenium<4.25.0 if needed

### TensorFlow Version Conflicts
**Problem**: TensorFlow 2.13 has strict typing-extensions requirements

**Solutions**:
1. Python 3.9 and earlier: Use `bernn[ml-full]` (includes TensorFlow constraints)
2. Python 3.10+: Use newer TensorFlow with `typing-extensions>=4.9.0`
3. Separate ML environment: Create dedicated environment for deep learning

### Python 3.8 Specific Conflicts
**Problem**: Python 3.8 with modern packages (emoji, pydantic-core, selenium) requiring newer typing-extensions, but TensorFlow 2.13 requires older versions

**Solutions**:
1. **Recommended**: Use `pip install bernn[python38-full]` (PyTorch only, no TensorFlow conflicts)
2. **With TensorFlow**: Use `pip install bernn[python38-tensorflow]` (includes TensorFlow, may show warnings)
3. **Automated**: Run `python resolve_python38_conflicts.py` for guided resolution
4. **Manual**: Install PyTorch only: `pip install bernn[python38-ml-minimal]`
5. **Clean environment**: Create dedicated Python 3.8 environment
