from imports import*

corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0,1.0], [0.0, 1.0]])
a1 = corners[1, :] - corners[0, :]  
a2 = corners[3, :] - corners[0, :]  
gdim = 2 
fdim = 1
h = 0.02 #0.0178 p_m
nx = int(1.0 / h)  
ny = int(1.0 / h)   
gmsh.initialize()

occ = gmsh.model.occ
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if model_rank == 0:
   points = [occ.add_point(*corner, 0) for corner in corners]
   lines = [occ.add_line(points[i], points[(i + 1) % 4]) for i in range(4)]
   loop = occ.add_curve_loop(lines)
   unit_cell = occ.add_plane_surface([loop])
   vol_dimTag = (gdim, unit_cell)
   
   occ.synchronize()
    
   gmsh.model.mesh.setTransfiniteCurve(lines[0], nx + 1) 
   gmsh.model.mesh.setTransfiniteCurve(lines[2], nx + 1) 
   gmsh.model.mesh.setTransfiniteCurve(lines[1], ny + 1) 
   gmsh.model.mesh.setTransfiniteCurve(lines[3], ny + 1)
   
   gmsh.model.mesh.setTransfiniteSurface(unit_cell, arrangement="Alternate")
   
   occ.synchronize()
   
   gmsh.model.addPhysicalGroup(gdim, [vol_dimTag[1]], 1, name="Matrix")
   gmsh.model.addPhysicalGroup(fdim, [lines[0]], 1, name="bottom")
   gmsh.model.addPhysicalGroup(fdim, [lines[1]], 2, name="right")
   gmsh.model.addPhysicalGroup(fdim, [lines[2]], 3, name="top")
   gmsh.model.addPhysicalGroup(fdim, [lines[3]], 4, name="left")
    
   gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
   gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

   gmsh.model.mesh.generate(gdim)

   result = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
   domain = result[0]
   cells = result[1]
   facets = result[2]

   gmsh.finalize()

vol = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=domain)))
    
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

point_dof = fem.locate_dofs_geometrical(
                V, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
              )
bcs = [fem.dirichletbc(np.zeros((gdim,)), point_dof, V)]

def periodic_relation_left_right(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0] - a1[0]
    out_x[1] = x[1] - a1[1]
    out_x[2] = x[2]
    return out_x

def periodic_relation_bottom_top(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0] - a2[0]
    out_x[1] = x[1] - a2[1]
    out_x[2] = x[2]
    return out_x

mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_periodic_constraint_topological(
    V, facets, 2, periodic_relation_left_right, bcs
    )
mpc.create_periodic_constraint_topological(
    V, facets, 3, periodic_relation_bottom_top, bcs
    )
mpc.finalize()

V1 = fem.functionspace(domain, ("Lagrange", 1))

mpc1 = dolfinx_mpc.MultiPointConstraint(V1)
mpc1.create_periodic_constraint_topological(
    V1, facets, 2, periodic_relation_left_right, bcs
    )
mpc1.create_periodic_constraint_topological(
    V1, facets, 3, periodic_relation_bottom_top, bcs
    )
mpc1.finalize()