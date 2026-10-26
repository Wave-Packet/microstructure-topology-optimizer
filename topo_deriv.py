from C_hom_calcul import*

V_tensor3 = fem.functionspace(domain, ("DG", 0, (3, 3)))  
"""V_tensor2 = fem.functionspace(domain,("DG", 0, (2,2)))

epsilon_u = fem.Function(V_tensor2)
sigma_v = fem.Function(V_tensor2)"""
DC_hom = fem.Function(V_tensor3)
Comp = fem.Function(V0)
DJ = fem.Function(V0)

#m = len(sigma_v.x.array) // 4
n = len(DC_hom.x.array) // 9

DC_hom_tens = DC_hom.x.array.reshape(n, 3, 3)

def DC_hom_Calcul():

    for nload in range(dim_load):
    
        Eps.value = elementary_load[nload]
        Eps_1.value = elementary_load[nload]
        u = problem_u.solve()
        """epsilon_u_expr = Expression(epsilon(u, Eps_1), V_tensor2.element.interpolation_points)
        epsilon_u.interpolate(epsilon_u_expr)
        epsilon_u_tens = epsilon_u.x.array.reshape(m, gdim, gdim)"""
        for nload_ in range(dim_load):
            Eps_.value = elementary_load[nload_]           
            Eps.value = elementary_load[nload_]
            v = problem_v.solve()
            #C1 = -E.x.array * ((1.0 - gamma.x.array)/(1.0 + alpha * gamma.x.array)) * 4.0
            #C2 = E.x.array * ((1.0 - gamma.x.array)/(1.0 + alpha * gamma.x.array)) * ((1.0 - gamma.x.array * (alpha - 2.0 * beta))/(1.0 + beta * gamma.x.array))
            C1 = -(1 - gamma) * (1 + beta) / (1 + beta * gamma)
            C2 = -0.5 * ((1 - gamma) / (1 + beta * gamma)) * ((1 - gamma) / (1 + alpha * gamma)) * (alpha - beta)
            """sigma_v_expr = Expression(sigma(v, Eps_), V_tensor2.element.interpolation_points)
            sigma_v.interpolate(sigma_v_expr)
            sigma_v_tens = sigma_v.x.array.reshape(m, gdim, gdim)
            DC_hom_tens[:, nload, nload_] = (
                C1 * np.einsum('nij,nij->n', sigma_v_tens, epsilon_u_tens) 
                + C2 * np.einsum('nii,njj->n', sigma_v_tens, epsilon_u_tens)
                    )"""
            """DC_hom_tens[:, nload, nload_] = C1 * np.sum(sigma_u_tens * sigma_v_tens, axis = (1, 2)) + C2 * np.trace(sigma_u_tens, axis1 = 1, axis2 = 2) * np.trace(sigma_v_tens, axis1 = 1, axis2 =2)
            Comp_expr = C1 * ufl.inner(sigma(u, Eps_1, E1), sigma(U[nload_], Eps_, E1)) + C2 * ufl.tr(sigma(U[nload], Eps_1, E1)) * ufl.tr(sigma(U[nload_], Eps_, E1)) #ufl.inner(eta(sigma(u, Eps_1)), sigma(v, Eps_))
            Comp.interpolate(Expression(Comp_expr, V0.element.interpolation_points))"""
            Comp_expr = (
                C1 * ufl.inner(sigma(v, Eps_), epsilon(u, Eps_1)) 
                + C2 * ufl.tr(sigma(v, Eps_)) * ufl.tr(epsilon(u, Eps_1))
                )
            Comp.interpolate(Expression(Comp_expr, V0.element.interpolation_points))
            DC_hom_tens[:, nload, nload_] = Comp.x.array
    
    return DC_hom_tens       