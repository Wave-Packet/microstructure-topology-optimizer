from __future__ import annotations
from typing import List, Tuple

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem as fem
import numpy as np
import ufl
from dolfinx import default_scalar_type, plot
from dolfinx.mesh import create_unit_square, locate_entities_boundary, meshtags
from slepc4py import SLEPc
from ufl import TestFunction, TrialFunction

from dolfinx_mpc import MultiPointConstraint, assemble_matrix
import pyvista as pv


def print0(string: str):
    if MPI.COMM_WORLD.rank == 0:
        print(string)


def monitor_EPS_short(EPS: SLEPc.EPS, it: int, nconv: int, eig: list, err: list, it_skip: int):
    
    if it == 1:
        print0("******************************")
        print0("***  SLEPc Iterations...   ***")
        print0("******************************")
        print0("Iter. | Conv. | Max. error")
        print0(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")
    elif not it % it_skip:
        print0(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")


def EPS_print_results(EPS: SLEPc.EPS):
    print0("\n******************************")
    print0("*** SLEPc Solution Results ***")
    print0("******************************")
    its = EPS.getIterationNumber()
    print0(f"Iteration number: {its}")
    nconv = EPS.getConverged()
    print0(f"Converged eigenpairs: {nconv}")

    if nconv > 0:
        vr, vi = EPS.getOperators()[0].createVecs()
        print0("\nConverged eigval.  Error ")
        print0("----------------- -------")
        for i in range(nconv):
            k = EPS.getEigenpair(i, vr, vi)
            error = EPS.computeError(i)
            if not np.isclose(k.imag, 0.0):
                print0(f" {k.real:2.2e} + {k.imag:2.2e}j {error:1.1e}")
            else:
                pad = " " * 11
                print0(f" {k.real:2.2e} {pad} {error:1.1e}")


def EPS_get_spectrum(
    EPS: SLEPc.EPS, mpc: MultiPointConstraint
) -> Tuple[List[complex], List[PETSc.Vec], List[PETSc.Vec]]: 
    
    eigval = [EPS.getEigenvalue(i) for i in range(EPS.getConverged())]
    eigvec_r = list()
    eigvec_i = list()
    V = mpc.function_space
    for i in range(EPS.getConverged()):
        vr = fem.Function(V)
        vi = fem.Function(V)

        EPS.getEigenvector(i, vr.x.petsc_vec, vi.x.petsc_vec)
        eigvec_r.append(vr)
        eigvec_i.append(vi) 
    idx = np.argsort(np.real(np.array(eigval)), axis=0)
    eigval = [eigval[i] for i in idx]
    eigvec_r = [eigvec_r[i] for i in idx]
    eigvec_i = [eigvec_i[i] for i in idx]
    return (eigval, eigvec_r, eigvec_i)


def solve_GEP_shiftinvert(
    A: PETSc.Mat,  
    B: PETSc.Mat,  
    problem_type: SLEPc.EPS.ProblemType = SLEPc.EPS.ProblemType.GNHEP,
    solver: SLEPc.EPS.Type = SLEPc.EPS.Type.KRYLOVSCHUR,
    nev: int = 10,
    tol: float = 1e-7,
    max_it: int = 10,
    target: float = 0.0,
    shift: float = 0.0,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> SLEPc.EPS:
    
    EPS = SLEPc.EPS()
    EPS.create(comm=comm)
    EPS.setOperators(A, B)
    EPS.setProblemType(problem_type)
    
    EPS.setDimensions(nev=nev)

    EPS.setType(solver)
   
    EPS.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE) 
    EPS.setTarget(target)  
    EPS.setTolerances(tol=tol, max_it=max_it)
    ST = EPS.getST()
    ST.setType(SLEPc.ST.Type.SINVERT)
    ST.setShift(shift)
    EPS.setST(ST)
    it_skip = 1
    EPS.setMonitor(lambda eps, it, nconv, eig, err: monitor_EPS_short(eps, it, nconv, eig, err, it_skip))
    EPS.setFromOptions()
    EPS.solve()
    EPS_print_results(EPS)
    return EPS

def plot_mode(V, eigvec, scale=1.0, plot_mesh=True, title=None):
   
    u_mode = fem.Function(V)
    if isinstance(eigvec, np.ndarray):
        u_mode.x.array[:] = eigvec
    else:
        u_mode.x.array[:] = eigvec.x.array

    Vu = u_mode.function_space
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(Vu)
    u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    
    u_3D = np.zeros((u_geometry.shape[0], 3))
    u_3D[:, :2] = u_mode.x.array.reshape(-1, 2)
    u_grid.point_data[u_mode.name] = u_3D
    u_grid.set_active_vectors(u_mode.name)
    
    warped = u_grid.warp_by_vector(u_mode.name, factor=scale)

    plotter = pv.Plotter()
    plotter.window_size = (800, 300)
    plotter.add_mesh(warped)
    if plot_mesh:
        edges = warped.extract_all_edges()
        plotter.add_mesh(edges, color="k", line_width=1, opacity=0.5)
    plotter.view_xy()
    if title:
        plotter.add_text(title, font_size=14)
    plotter.show_axes()
    plotter.view_xy()
    plotter.show()

def assemble_and_solve(Nev: int = 10):
    comm = MPI.COMM_WORLD
    N = 250
    mesh = create_unit_square(comm, N, N)
    gdim = mesh.geometry.dim
    V = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    fdim = mesh.topology.dim - 1

    bcs = []
    pbc_directions = []
    pbc_slave_tags = []
    pbc_is_slave = []
    pbc_is_master = []
    pbc_meshtags = []
    pbc_slave_to_master_maps = []

    def generate_pbc_slave_to_master_map(i):
        def pbc_slave_to_master_map(x):
            out_x = x.copy()
            out_x[i] = x[i] - 1
            return out_x

        return pbc_slave_to_master_map

    def generate_pbc_is_slave(i):
        return lambda x: np.isclose(x[i], 1)

    def generate_pbc_is_master(i):
        return lambda x: np.isclose(x[i], 0)
    
    for i in range(2):
            
        pbc_directions.append(i)
        pbc_slave_tags.append(i + 2)
        pbc_is_slave.append(generate_pbc_is_slave(i))
        pbc_is_master.append(generate_pbc_is_master(i))
        pbc_slave_to_master_maps.append(generate_pbc_slave_to_master_map(i))  
        facets = locate_entities_boundary(mesh, fdim, pbc_is_slave[-1])
        arg_sort = np.argsort(facets)
        pbc_meshtags.append(
            meshtags(
                mesh,
                fdim,
                facets[arg_sort],
                np.full(len(facets), pbc_slave_tags[-1], dtype=np.int32),
            )
        )

    mpc = MultiPointConstraint(V)
    N_pbc = len(pbc_directions)
    for i in range(N_pbc):

        def pbc_slave_to_master_map(x):
            out_x = pbc_slave_to_master_maps[i](x)
            idx = pbc_is_slave[(i + 1) % N_pbc](x)
            out_x[pbc_directions[i]][idx] = np.nan
            return out_x

        mpc.create_periodic_constraint_topological(V, pbc_meshtags[i], pbc_slave_tags[i], pbc_slave_to_master_map, bcs)
    
    def pbc_slave_to_master_map(x):
        out_x = x.copy()
        out_x[0] = x[0] - 1
        out_x[1] = x[1] - 1
        idx = np.logical_and(pbc_is_slave[0](x), pbc_is_slave[1](x))
        out_x[0][~idx] = np.nan
        out_x[1][~idx] = np.nan
        return out_x

    mpc.create_periodic_constraint_topological(V, pbc_meshtags[1], pbc_slave_tags[1], pbc_slave_to_master_map, bcs)
    mpc.finalize()

    u = TrialFunction(V)
    v = TestFunction(V)
    rho = 1.0
    nu = 0.3
    #E = 2000
    #lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    #mu = E / (2 * (1 + nu))
    def phi_function(y):
        return (y[0] - 0.5)**2 + (y[1] - 0.5)**2 - 1/16

    def assign_E_based_on_phi(mesh, phi_function, E_positive, E_negative):
        V0 = fem.functionspace(mesh, ("DG", 0))
        y = ufl.SpatialCoordinate(mesh)
        phi_expr = phi_function(y)
        phi_vals = fem.Function(V0)
        phi_vals.interpolate(fem.Expression(phi_expr, V0.element.interpolation_points))
        E = fem.Function(V0, name="YoungModulus")
        E.x.array[:] = np.where(phi_vals.x.array > 0, E_positive, E_negative)
        return E

    def compute_lame_parameters(E, nu):
        V0 = E.function_space
        mu = fem.Function(V0)
        lmbda = fem.Function(V0)
        mu.x.array[:] = E.x.array / (2.0 * (1 + nu))
        lmbda.x.array[:] = E.x.array * nu / ((1 + nu) * (1 - 2 * nu))
        return mu, lmbda

    E_positive, E_negative = 2000, 50
    E = assign_E_based_on_phi(mesh, phi_function, E_positive, E_negative)
    mu, lmbda = compute_lame_parameters(E, nu)


    def eps(u):
       return ufl.sym(ufl.grad(u))

    def sigma(u):
       return lmbda * ufl.tr(eps(u)) * ufl.Identity(gdim) + 2.0 * mu * eps(u)

    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    b = rho * ufl.inner(u, v) * ufl.dx
    mass_form = fem.form(a)
    stiffness_form = fem.form(b)

    diagval_A = 1e5
    diagval_B = 1e-3
    tol = float(5e2 * np.finfo(default_scalar_type).resolution)

    A = assemble_matrix(mass_form, mpc, bcs=bcs, diagval=diagval_A)
    B = assemble_matrix(stiffness_form, mpc, bcs=bcs, diagval=diagval_B)
    EPS = solve_GEP_shiftinvert(
        A,
        B,
        problem_type=SLEPc.EPS.ProblemType.GHEP,
        solver=SLEPc.EPS.Type.KRYLOVSCHUR,
        nev=Nev,
        tol=tol,
        max_it=30,
        target=1.5,
        shift=1.5,
        comm=comm,
    )
    (eigval, eigvec_r, eigvec_i) = EPS_get_spectrum(EPS, mpc)
    for i in range(len(eigval)):
        eigvec_r[i].x.scatter_forward()
        mpc.backsubstitution(eigvec_r[i])
        eigvec_i[i].x.scatter_forward()
        mpc.backsubstitution(eigvec_i[i])
    print0(f"Computed eigenvalues:\n {np.around(eigval, decimals=2)}")

    if MPI.COMM_WORLD.rank == 0:
        for i, (vec, val) in enumerate(zip(eigvec_r[:10], eigval[:10])):
            plot_mode(
                V, 
                vec, 
                scale=0.01,
                title=f"Mode {i} (λ={val:.2f})",
                plot_mesh=True
            )   


assemble_and_solve(5)

