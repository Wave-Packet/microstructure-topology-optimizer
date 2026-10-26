from space_param_level_set import*

Eps = fem.Constant(domain, np.zeros((2, 2)))
Eps_ = fem.Constant(domain, np.zeros((2, 2)))
Eps_0 = fem.Constant(domain, np.zeros((2, 2)))
Eps_1 = fem.Constant(domain, np.zeros((2, 2)))

def epsilon(v, Eps = Eps_0):
    return ufl.sym(ufl.grad(v)) + Eps

def sigma(v, Eps = Eps_0,  E = E):
    eps = Eps + epsilon(v)
    return (E * nu / (1 + nu) / (1 - nu))* ufl.tr(eps) * ufl.Identity(gdim) + (E / (1 + nu)) * eps

du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
a_form, L_form = ufl.system(ufl.inner(sigma(du, Eps), epsilon(u_)) * ufl.dx)

u = fem.Function(mpc.function_space, name="Periodic_fluctuation")
v = fem.Function(mpc.function_space, name="Periodic_fluctuation")
u_d = fem.Function(mpc.function_space, name="Displacement")
v_d = fem.Function(mpc.function_space, name="Displacement")
#u3 = fem.Function(mpc.function_space, name="Periodic_fluctuation")
#v = fem.Function(mpc.function_space, name="iteration_function")
U = []
problem_v = LinearProblem(
    a_form,
    L_form,
    mpc,
    bcs=bcs,
    u=v,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

problem_u = LinearProblem(
    a_form,
    L_form,
    mpc,
    bcs=bcs,
    u=u,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

elementary_load = [
    np.array([[1.0, 0.0], [0.0, 0.0]]),
    np.array([[0.0, 0.0], [0.0, 1.0]]),
    np.array([[0.0, 0.5], [0.5, 0.0]]),
    ]
load_labels = ["Exx", "Eyy", "Exy"]
dim_load = len(elementary_load)
def aux_prob_updt():
    for nload in range(dim_load):
        Eps.value = elementary_load[nload]
        problem_u.solve()
        U.append(u)
    return U

def aux_prob_init():
    U = []
    return U
        
def plot_warped(u, scale=1.0, plot_mesh=False, user_callback=None, title=None):
    Vu = u.function_space
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(Vu)
    u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_3D = np.zeros((u_geometry.shape[0], 3))
    u_3D[:, :2] = u.x.array.reshape(-1, 2)
    u_grid.point_data[u.name] = u_3D
    u_grid.set_active_vectors(u.name)
    warped = u_grid.warp_by_vector(u.name, factor=scale)

    plotter = pv.Plotter()
    plotter.window_size = (800, 300)
    plotter.add_mesh(warped)
    if plot_mesh:
        edges = warped.extract_all_edges()
        plotter.add_mesh(edges, color="k", line_width=1, opacity=0.5)
    plotter.view_xy()
    if title:
        plotter.add_text(title, font_size=14)
    plotter.show()