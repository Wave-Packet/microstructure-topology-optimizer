import os

from imports import*

comm = MPI.COMM_WORLD
N = int(os.environ.get("MICROTOPOLOGY_MESH_N", "62"))
uj_flag = dolfinx.cpp.mesh.DiagonalType.left_right
crossed = dolfinx.cpp.mesh.DiagonalType.crossed
domain = create_unit_square(comm, N, N, diagonal = uj_flag)
#h = domain.h(domain.topology.dim ,domain.topology.index_map(domain.topology.dim).owners)
gdim = domain.geometry.dim
V = fem.functionspace(domain, ("P", 1, (gdim,)))
V1 = fem.functionspace(domain, ("P", 1))
fdim = domain.topology.dim - 1
vol = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=domain)))

cells = domain.topology.index_map(domain.topology.dim).size_global
nodes = domain.geometry.x.shape[0]

print(f"cells : {cells}, nodes : {nodes}")

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
    facets = locate_entities_boundary(domain, fdim, pbc_is_slave[-1])
    arg_sort = np.argsort(facets)
    pbc_meshtags.append(
        meshtags(
          domain,
          fdim,
          facets[arg_sort],
          np.full(len(facets), pbc_slave_tags[-1], dtype=np.int32),
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

