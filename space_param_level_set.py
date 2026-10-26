from maillages_Walid import*

nu = 0.3

V0 = fem.functionspace(domain, ("DG", 0))
V_tensor3 = fem.functionspace(domain, ("DG", 0, (3, 3)))    
def phi_function(x):
    return ((ufl.cos(ufl.pi * (x[0] - 0.5)))**2.0) * ((ufl.cos(ufl.pi * (x[1] - 0.5)))**2.0) - 0.5
    #return (y[0] - 0.5)**2 + (y[1] - 0.5)**2 - 1/16

y = ufl.SpatialCoordinate(domain)

phi_expr = phi_function(y)
phi_vals = fem.Function(mpc1.function_space)
phi_new = fem.Function(mpc1.function_space)
phi_old = fem.Function(mpc1.function_space)
phi_vals.interpolate(fem.Expression(phi_expr, mpc1.function_space.element.interpolation_points))
phiNorm2 = assemble_scalar(form(ufl.inner(phi_vals, phi_vals) * ufl.dx))
phi_vals.x.array[:] = (1/np.sqrt(phiNorm2)) * phi_vals.x.array

E_positive, E_negative = 0.01, 1.0
rho = fem.Function(mpc1.function_space)
density = fem.Function(V0)
rho.x.array[:] = np.where(phi_vals.x.array < 0, 1.0, 0.0)
E1 = fem.Function(V0)
E = fem.Function(V0, name = "YoungModulus")
E1.x.array[:] = np.ones(len(E1.x.array))
density.interpolate(rho)
conditions = [density.x.array == 0.0, density.x.array == 1.0 / 3.0, density.x.array == 2.0 / 3.0, density.x.array == 1.0]
val_E = [E_positive, (1.0 / 3.0) * (E_negative - E_positive) + E_positive, (2.0 / 3.0) * (E_negative - E_positive) + E_positive, E_negative]
E.x.array[:] = np.select(conditions, val_E, density.x.array)
#E.x.array[:] = np.where(phi_vals.x.array < 0, E_negative, E_positive)
gamma_star = 0.01
gamma = fem.Function(V0)
val_gamma = [1.0 / gamma_star, (1.0 / 3.0) * (gamma_star - 1.0 / gamma_star) + 1.0 / gamma_star, (2.0 / 3.0) * (gamma_star - 1.0 / gamma_star) + 1.0 / gamma_star , gamma_star]
gamma.x.array[:] = np.select(conditions, val_gamma, density.x.array)
#gamma.x.array[:] = np.where(phi_vals.x.array < 0, gamma_star, 1 / gamma_star)
alpha = (1 + nu)/(1 - nu)
beta = (3 - nu)/(1 + nu)

def area_updt():
    chi = ufl.conditional(ufl.lt(phi_vals, 0.0), 1.0, 0.0)
    dx = ufl.dx(domain=domain)
    area_local = fem.assemble_scalar(fem.form(chi * dx))
    return area_local
area1 = area_updt() 

"""with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

E.name = "Y"
with io.XDMFFile(MPI.COMM_WORLD, "solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(E, 0.0) # Fonction + pas de temps
"""
