"""
Backwards-compatibility shim.

Use `humanbench run [model]` instead.
Install once with:  pip install -e .
"""
from humanbench.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
