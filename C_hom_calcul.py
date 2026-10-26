from auxil_prob_BW import*

C_hom = np.zeros((dim_load, dim_load))

def C_hom_calcul():
        
    for nload in range(dim_load):
        Eps.value = elementary_load[nload]
        u = problem_u.solve()
                
        for nload_ in range(dim_load):
            Eps_.value = elementary_load[nload_]    
            C_hom[nload, nload_] = (
                  fem.assemble_scalar(fem.form(ufl.inner(sigma(u,Eps), Eps_) * ufl.dx)) / vol
                  )
    
    return C_hom

