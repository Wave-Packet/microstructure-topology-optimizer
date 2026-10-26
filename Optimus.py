"""Level-set topology optimization of a periodic microstructure."""

from __future__ import annotations

import argparse
import os

from criteria import CRITERIA, DEFAULT_CRITERION, criteria_table, get_criterion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--criterion",
        choices=tuple(CRITERIA),
        default=DEFAULT_CRITERION,
        help="fonction objectif a minimiser",
    )
    parser.add_argument(
        "--volume-penalty",
        type=float,
        default=None,
        help="remplace la penalite volumique par defaut du critere",
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--mesh-resolution",
        type=int,
        default=62,
        help="nombre de subdivisions par direction (defaut: 62)",
    )
    parser.add_argument("--no-plot", action="store_true", help="desactive PyVista")
    parser.add_argument("--list-criteria", action="store_true")
    args = parser.parse_args()
    if args.max_iterations < 1:
        parser.error("--max-iterations doit etre superieur ou egal a 1")
    if args.mesh_resolution < 2:
        parser.error("--mesh-resolution doit etre superieur ou egal a 2")
    return args


args = parse_args()
if args.list_criteria:
    print(criteria_table())
    raise SystemExit(0)

# This must be set before importing the original module chain, which creates
# the mesh at import time.
os.environ["MICROTOPOLOGY_MESH_N"] = str(args.mesh_resolution)

from C_hom_calcul import *  # noqa: E402,F403
from topo_deriv import *  # noqa: E402,F403


criterion = get_criterion(args.criterion)
volume_penalty = (
    criterion.default_volume_penalty
    if args.volume_penalty is None
    else args.volume_penalty
)

print("Volume:", vol)
print(f"Critere: {criterion.name} (equation {criterion.equation})")
print(f"Penalite volumique: {volume_penalty:g}")

theta_tol = 1.0 * np.pi / 180
tau_tol = 0.001
tau = 1.0

DJ_sgn = fem.Function(V0)
DJ_ls = fem.Function(mpc1.function_space)
w = ufl.TrialFunction(V1)
z = ufl.TestFunction(V1)

a = (1 / N**2) * ufl.inner(ufl.grad(w), ufl.grad(z)) * ufl.dx + ufl.inner(w, z) * ufl.dx
L = ufl.inner(DJ_sgn, z) * ufl.dx
DJ_filter = fem.petsc.LinearProblem(a, L, u=DJ_ls, petsc_options_prefix="proj_")

for i in range(args.max_iterations):
    C_hom = C_hom_calcul()
    lmbda_hom = C_hom[0, 1]
    mu_hom = C_hom[2, 2]
    nu_hom = lmbda_hom / (lmbda_hom + mu_hom) / 2
    DC_hom_tens = DC_hom_Calcul()

    area1 = area_updt()
    objective, objective_gradient = criterion.evaluate(C_hom, DC_hom_tens)
    Jn = objective + volume_penalty * area1 / vol
    DJ.x.array[:] = objective_gradient

    DJ_sgn.interpolate(
        fem.Expression(
            -ufl.inner(rho, DJ) + ufl.inner((1 - rho), DJ),
            V0.element.interpolation_points,
        )
    )
    DJ_filter.solve()
    DJ_ls.x.array[:] += volume_penalty
    DJnorm2 = assemble_scalar(form(ufl.inner(DJ_ls, DJ_ls) * ufl.dx))
    DJ_ls.x.array[:] = (1.0 / np.sqrt(DJnorm2)) * DJ_ls.x.array

    psDJphi = assemble_scalar(form(ufl.inner(DJ_ls, phi_vals) * ufl.dx))
    psDJphi = np.clip(psDJphi, -1.0, 1.0)
    theta = np.arccos(psDJphi)

    print("iteration", i, "aire", area1)
    print(C_hom)
    print("theta_deg", theta * 180 / np.pi, "J", Jn, "nu_hom", nu_hom)

    if theta < theta_tol:
        break

    tau = min(1, 1.5 * tau)
    phi_old.x.array[:] = phi_vals.x.array
    Jold = Jn
    Jnew = Jold + 1.0

    while Jnew > Jold:
        if tau < tau_tol / 2:
            phi_vals.x.array[:] = phi_old.x.array
            Jn = Jold
            break

        phi_new.x.array[:] = (1.0 / np.sin(theta)) * (
            np.sin((1.0 - tau) * theta) * phi_old.x.array
            + np.sin(tau * theta) * DJ_ls.x.array
        )
        phiNorm2 = assemble_scalar(form(ufl.inner(phi_new, phi_new) * ufl.dx))
        phi_new.x.array[:] = (1 / np.sqrt(phiNorm2)) * phi_new.x.array
        phi_vals.x.array[:] = phi_new.x.array

        rho.x.array[:] = np.where(phi_vals.x.array < 0, 1.0, 0.0)
        density.interpolate(rho)
        conditions = [
            density.x.array == 0.0,
            density.x.array == 1.0 / 3.0,
            density.x.array == 2.0 / 3.0,
            density.x.array == 1.0,
        ]
        E.x.array[:] = np.select(conditions, val_E, density.x.array)
        gamma.x.array[:] = np.select(conditions, val_gamma, density.x.array)
        area1 = area_updt()

        C_hom = C_hom_calcul()
        objective, _ = criterion.evaluate(C_hom, DC_hom_tens)
        Jn = objective + volume_penalty * area1 / vol
        Jnew = Jn
        tau = tau / 2
        print("tatonnements", tau)

    tau = 2 * tau

    with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)

    E.name = "Y"
    with io.XDMFFile(MPI.COMM_WORLD, "solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(E, 0.0)

    if not args.no_plot:
        topology, cell_types, geometry = plot.vtk_mesh(domain)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        grid.cell_data["YoungModulus"] = E.x.array
        plotter = pv.Plotter()
        plotter.add_mesh(
            grid,
            scalars="YoungModulus",
            color="red",
            cmap="plasma",
            opacity=0.8,
            show_edges=True,
        )
        plotter.view_xy()
        plotter.show()

print("theta =", theta, "tau =", tau, "J =", Jn, "area =", area1)
