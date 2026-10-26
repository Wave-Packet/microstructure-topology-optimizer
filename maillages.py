from imports import*

comm = MPI.COMM_WORLD
N = 55
domain = create_unit_square(comm, N, N)
gdim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
fdim = domain.topology.dim - 1
vol = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=domain)))

cells = domain.topology.index_map(domain.topology.dim).size_global
nodes = domain.geometry.x.shape[0]

print(f"cells : {cells}, nodes : {nodes}")

def bottom_entities_marker(x):
    return np.isclose(x[1], 0.0) 
    
def right_entities_marker(x):
    return np.isclose(x[0], 1.0) 

def top_entities_marker(x):       
    return np.isclose(x[1], 1.0)

def left_entities_marker(x):
    return np.isclose(x[0], 0.0) 
        
bottom_entities_index = locate_entities_boundary(domain, fdim, bottom_entities_marker)
right_entities_index = locate_entities_boundary(domain, fdim, right_entities_marker)
top_entities_index = locate_entities_boundary(domain, fdim, top_entities_marker)
left_entities_index = locate_entities_boundary(domain, fdim, left_entities_marker)

bottom_entities_value = np.full_like(bottom_entities_index, 1, dtype=np.int32)
right_entities_value = np.full_like(right_entities_index, 2, dtype=np.int32)
top_entities_value = np.full_like(top_entities_index, 3, dtype=np.int32)
left_entities_value = np.full_like(left_entities_index, 4, dtype=np.int32)

all_indices = np.hstack([
    bottom_entities_index, 
    right_entities_index, 
    top_entities_index, 
    left_entities_index
])

all_values = np.hstack([
    bottom_entities_value, 
    right_entities_value, 
    top_entities_value, 
    left_entities_value
])

sort_order = np.argsort(all_indices)

all_indices_sorted = all_indices[sort_order].astype(np.int32)
all_values_sorted = all_values[sort_order].astype(np.int32)

facets = meshtags(domain, fdim, all_indices_sorted, all_values_sorted)

V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

point_dof = fem.locate_dofs_geometrical(
                V, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0)
              )
bcs = [fem.dirichletbc(np.zeros((gdim,)), point_dof, V)]

def periodic_relation_left_right(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0] - 1.0
    out_x[1] = x[1] - 0.0
    out_x[2] = x[2]
    return out_x

def periodic_relation_bottom_top(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0] - 0.0
    out_x[1] = x[1] - 1.0
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

with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)