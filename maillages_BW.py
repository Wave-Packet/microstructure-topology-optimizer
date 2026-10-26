from imports import*

corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0,1.0], [0.0, 1.0]])
a1 = corners[1, :] - corners[0, :]  
a2 = corners[3, :] - corners[0, :]  
gdim = 2 
fdim = 1
h = 0.018
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

V = fem.functionspace(domain, ("P", 1, (gdim,)))
V1 = fem.functionspace(domain, ("P", 1))

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
    #facets = locate_entities_boundary(domain, fdim, pbc_is_slave[-1])
    arg_sort = np.argsort(facets.indices)
    pbc_meshtags.append(
        meshtags(
          domain,
          fdim,
          facets.indices[arg_sort],
          np.full(len(facets.indices), pbc_slave_tags[-1], dtype=np.int32),
          )
    )

mpc = MultiPointConstraint(V)
mpc1 = MultiPointConstraint(V1)
N_pbc = len(pbc_directions)
for i in range(N_pbc):

    def pbc_slave_to_master_map(x):
        out_x = pbc_slave_to_master_maps[i](x)
        idx = pbc_is_slave[(i + 1) % N_pbc](x)
        out_x[pbc_directions[i]][idx] = np.nan
        return out_x

    mpc.create_periodic_constraint_topological(V, pbc_meshtags[i], pbc_slave_tags[i], pbc_slave_to_master_map, bcs)
    mpc1.create_periodic_constraint_topological(V1, pbc_meshtags[i], pbc_slave_tags[i], pbc_slave_to_master_map, bcs)
def pbc_slave_to_master_map(x):
    out_x = x.copy()
    out_x[0] = x[0] - 1
    out_x[1] = x[1] - 1
    idx = np.logical_and(pbc_is_slave[0](x), pbc_is_slave[1](x))
    out_x[0][~idx] = np.nan
    out_x[1][~idx] = np.nan
    return out_x

mpc.create_periodic_constraint_topological(V, pbc_meshtags[1], pbc_slave_tags[1], pbc_slave_to_master_map, bcs)
mpc1.create_periodic_constraint_topological(V1, pbc_meshtags[1], pbc_slave_tags[1], pbc_slave_to_master_map, bcs)

mpc.finalize()
mpc1.finalize()        
