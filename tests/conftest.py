import os
import sys


os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def pytest_sessionstart(session):
    # Ensure imports resolve to the workspace package, not an installed wheel.
    for name in list(sys.modules.keys()):
        if name == "bernn" or name.startswith("bernn."):
            module = sys.modules.get(name)
            module_file = getattr(module, "__file__", "") or ""
            if "site-packages" in module_file:
                sys.modules.pop(name, None)
