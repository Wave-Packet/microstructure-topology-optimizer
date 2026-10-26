from __future__ import annotations
import numpy as np
from typing import List, Tuple
from mpi4py import MPI
import gmsh
import ufl
from dolfinx import fem, io, plot, default_scalar_type, la
import dolfinx.fem.petsc
from dolfinx.io.gmsh import model_to_mesh
import dolfinx_mpc.utils
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx.fem import Expression, assemble_scalar, form, locate_dofs_topological
from dolfinx.mesh import create_unit_square, locate_entities_boundary, meshtags, refine
import pyvista as pv
from petsc4py import PETSc
from slepc4py import SLEPc


