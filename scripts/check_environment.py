"""Fail fast when a required scientific dependency cannot be imported."""

from __future__ import annotations

import importlib


MODULES = (
    "numpy",
    "mpi4py",
    "petsc4py",
    "slepc4py",
    "ufl",
    "gmsh",
    "dolfinx",
    "dolfinx_mpc",
    "pyvista",
)


def main() -> None:
    for name in MODULES:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "version non exposee")
        print(f"{name:12s} OK  {version}")


if __name__ == "__main__":
    main()
