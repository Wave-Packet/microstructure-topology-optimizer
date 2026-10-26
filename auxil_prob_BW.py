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

problem_v = LinearProblem(
    a_form,
    L_form,
    mpc,
    bcs=bcs,
    u=v,
    petsc_options={"ksp_type": "preonly",
      "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps"},#"ksp_type": "cg", "pc_type": "gamg"},
    )

problem_u = LinearProblem(
    a_form,
    L_form,
    mpc,
    bcs=bcs,
    u=u,
    petsc_options={ "ksp_type": "preonly",
      "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps"},#"ksp_type": "cg", "pc_type": "gamg"},
    )

elementary_load = [
    np.array([[1.0, 0.0], [0.0, 0.0]]),
    np.array([[0.0, 0.0], [0.0, 1.0]]),
    np.array([[0.0, 0.5], [0.5, 0.0]]),
    ]
load_labels = ["Exx", "Eyy", "Exy"]
dim_load = len(elementary_load)
