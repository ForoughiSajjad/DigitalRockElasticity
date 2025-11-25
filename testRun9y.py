import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.mesh import CellType, meshtags
from dolfinx.fem import Function, functionspace, dirichletbc, Constant, Expression
from ufl import Identity, sym, nabla_grad, tr, dx, inner
from dolfinx.fem.petsc import LinearProblem
from skimage import io as skimage_io
from skimage.transform import downscale_local_mean
from petsc4py import PETSc
import time
import sys
import gc  # For memory management
from dolfinx.cpp.mesh import CellType, GhostMode, create_cell_partitioner
import psutil
import resource  # For memory tracking on Linux
from skimage.measure import block_reduce

def print_memory_usage(comm, message):
    process = psutil.Process()
    mem = process.memory_info().rss/1024**3 # Use 1024**3 for GiB
    if comm.rank == 0:
        print(f"{message}: {mem:.2f} GiB (Rank 0)")
    comm.Barrier()

def boxcount_3d(Z, k):
    # Reduce the 3D array by taking the maximum in non-overlapping kxkxk blocks
    S = block_reduce(Z, block_size=(k, k, k), func=np.max)
    return np.sum(S > 0)

def fractal_dimension_3d(Z, min_box=2, max_box=None):
    if max_box is None:
        max_box = min(Z.shape) // 2
    # Ensure Z is binary
    Z = (Z > 0).astype(np.uint8)
    # Use power-of-two box sizes
    sizes = 2 ** np.arange(int(np.log2(max_box)), int(np.log2(min_box)) - 1, -1)
    counts = [boxcount_3d(Z, size) for size in sizes]
    # Remove zero counts to avoid log(0)
    counts = np.array(counts)
    sizes = sizes[counts > 0]
    counts = counts[counts > 0]
    # Fit log-log data to find slope
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dim = -coeffs[0]
    return fractal_dim

def get_peak_memory_MB():
    """Return the peak resident set size (physical memory) in MB."""
    try:
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # in KB
        if comm.rank == 0:
            print(f"Current memory usage: {peak_kb/1024:.2f} MB (Rank 0)")
        return peak_kb / 1024  # convert to MB
    except Exception:
        process = psutil.Process()
        peak = process.memory_info().rss / 1024 / 1024
        if comm.rank == 0:
            print(f"Current memory usage: {peak:.2f} MB (Rank 0)")
        return peak

# Track peak memory usage
peak_memory = 0
def update_peak_memory():
    global peak_memory
    current_mem = get_peak_memory_MB()
    if current_mem is not None and current_mem > peak_memory:
        peak_memory = current_mem

def parse_input(filename, comm):
    """Parse input file with image parameters and material properties"""
    materials = {}
    image_name = None
    voxel_size = None
    coarsenLevel= None
    image_name = None
    uniaxial_direction = None
    poreLabel = None
    
    if comm.rank == 0:
        try:
            with open(filename, 'r') as f:
                line = f.readline().strip()
                if not line.startswith('image_name'):
                    raise ValueError("First  must specify image_name")
                image_name = line.split('=')[1].strip().strip('"')

                line = f.readline().strip()
                if not line.startswith('uniaxial_direction'):
                    raise ValueError("First line must specify image_name")
                uniaxial_direction = line.split('=')[1].strip().strip('"')

                line = f.readline().strip()
                if not line.startswith('voxel_size'):
                    raise ValueError("Second line must specify voxel_size")
                voxel_size = float(line.split('=')[1].strip())

                line = f.readline().strip()
                if not line.startswith('coarsening_level'):
                    raise ValueError("Second line must specify coarsening_level")
                coarsenLevel = float(line.split('=')[1].strip())                
                
                line = f.readline().strip()
                if not line.startswith('pore_label'):
                    raise ValueError("Third line must specify pore_label")
                poreLabel = int(line.split('=')[1].strip())

                header = f.readline().strip()
                if "phase" not in header.lower() or "k" not in header.lower() or "g" not in header.lower():
                    print(f"Warning: Unconventional header '{header}'", file=sys.stderr)

                for line_num, line in enumerate(f, start=4):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) != 3:
                        print(f"Warning: Skipping line {line_num}: '{line}'", file=sys.stderr)
                        continue
                    try:
                        phase = int(parts[0])
                        K = float(parts[1])
                        G = float(parts[2])
                        materials[phase] = (K, G)
                    except ValueError:
                        print(f"Warning: Bad line {line_num}: '{line}'", file=sys.stderr)
                        continue
            if not materials:
                raise ValueError("No valid materials found")

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found", file=sys.stderr)
            image_name, voxel_size, materials = None, None, None
        except Exception as e:
            print(f"Error parsing '{filename}': {e}", file=sys.stderr)
            image_name, voxel_size, materials = None, None, None

    error_signal = 0
    if comm.rank == 0 and materials is None:
        error_signal = 1
    error_signal = comm.bcast(error_signal, root=0)

    if error_signal:
        comm.Barrier()
        raise RuntimeError("Input parsing failed on rank 0")

    image_name = comm.bcast(image_name, root=0)
    voxel_size = comm.bcast(voxel_size, root=0)
    materials = comm.bcast(materials, root=0)
    
    # ADD THESE
    uniaxial_direction = comm.bcast(uniaxial_direction, root=0)
    coarsenLevel = comm.bcast(coarsenLevel, root=0)
    poreLabel = comm.bcast(poreLabel, root=0)
    
    return image_name, uniaxial_direction, voxel_size, coarsenLevel, poreLabel, materials

# --- Main Script 
start_time = time.time()
comm = MPI.COMM_WORLD

input_file = "input_y.dat"
downsample_factor = 1

solver_params = {
    "ksp_type": "cg",
    "ksp_rtol": 1e-6,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_coarsen_type": "HMIS",
    "pc_hypre_boomeramg_interp_type": "ext+i-cc",
    "pc_hypre_boomeramg_agg_nl": 2,
    "pc_hypre_boomeramg_truncfactor": 0.05,
    "pc_hypre_boomeramg_max_levels": 25,
    "mat_type": "aij",
    "ksp_max_it": 2000,
    "ksp_norm_type": "unpreconditioned",
    # Removed 'memory_optimized' as it's not a standard PETSc option
}

try:
    # Parse input file and distribute parameters
    image_name, uniaxial_direction, voxel_size_orig, coarsenLevel, poreLabel, materials  = parse_input(input_file, comm)
except RuntimeError:
    if comm.rank == 0:
        print("Exiting due to input error", file=sys.stderr)
    sys.exit(1)

if comm.rank == 0:
    print(f"Materials: {materials}")
    sys.stdout.flush()
# Compute actual voxel size accounting for any downsampling factor
voxel_size = voxel_size_orig * downsample_factor
# --- Image Loading & Broadcast ---
image_downsampled = None
nz, ny, nx = 0, 0, 0
exportPorosity = 0.0
fractalDImension = 0.0
# Load image on rank 0 and distribute to all ranks
if comm.rank == 0:
    print(f"Rank 0: Loading '{image_name}'...")
    sys.stdout.flush()
    try:
        image_full = skimage_io.imread(image_name)
        # Align image axes with x-y-z if needed
        if uniaxial_direction == 'z':
            image_full = image_full.transpose(2, 1, 0)
        elif uniaxial_direction == 'y':
            image_full = image_full.transpose(1, 0, 2)

        # Optional downsampling (if downsample_factor > 1)
        if downsample_factor > 1:
            trimmed_dims = (
                (image_full.shape[0] // downsample_factor) * downsample_factor,
                (image_full.shape[1] // downsample_factor) * downsample_factor,
                (image_full.shape[2] // downsample_factor) * downsample_factor,
            )
            image_trimmed = image_full[:trimmed_dims[0], :trimmed_dims[1], :trimmed_dims[2]]
            image_downsampled = downscale_local_mean(
                image_trimmed,
                (downsample_factor, downsample_factor, downsample_factor),
            ).astype(image_full.dtype)
            update_peak_memory()
            del image_full, image_trimmed
            gc.collect()
            print("Rank 0: Freed original image data")
        else:
            image_downsampled = image_full
            del image_full
            gc.collect()

        # Memory optimization: force image data to 8-bit
        if image_downsampled.dtype != np.uint8:
            print("Converting to uint8 for memory efficiency", file=sys.stderr)
            image_downsampled = np.around(image_downsampled).astype(np.uint8)
            # (Preserve any label values by rounding before casting)

        nz, ny, nx = image_downsampled.shape
        pore_volume = np.sum(image_downsampled == poreLabel)
        exportPorosity = pore_volume / (nx * ny * nz)
        fractalDImension = fractal_dimension_3d(image_downsampled != poreLabel)
        print(f"Downsampled dims: ({nz},{ny},{nx}), Porosity={exportPorosity:.4f}")

    except Exception as e:
        print(f"Image error: {e}", file=sys.stderr)
        comm.Abort(1)

# Broadcast image dimensions to all ranks
dims = np.array([nz, ny, nx], dtype=int)
comm.Bcast([dims, MPI.INT], root=0)
nz, ny, nx = dims

# Broadcast image data as contiguous uint8 buffer
authoritative_dtype = np.uint8
if comm.rank == 0:
    image_buf = np.ascontiguousarray(image_downsampled)
else:
    image_buf = np.empty((nz, ny, nx), dtype=authoritative_dtype)
comm.Bcast([image_buf, MPI.UNSIGNED_CHAR], root=0)
# All ranks now have image_buf
del image_downsampled
gc.collect()

# --- Mesh Creation ---
Lx = np.float32(nx * voxel_size)
Ly = np.float32(ny * voxel_size)
Lz = np.float32(nz * voxel_size)
# Use float32 for domain dimensions to reduce memory usage
if comm.rank == 0:
    print(f"Creating domain: {Lx:.2f} x {Ly:.2f} x {Lz:.2f}")
    sys.stdout.flush()

update_peak_memory()
# Define the mesh (single precision coordinates)
coarsenLevel = 2
def cells_after_coarsening(nx, ny, nz, factor, min_cells=2):
    # factor can be float (e.g., 1.5 means ~1.5× coarser; <1 means refine)
    n_new = (
        max(min_cells, int(round(nx / factor))),
        max(min_cells, int(round(ny / factor))),
        max(min_cells, int(round(nz / factor))),
    )
    return n_new

n0 = cells_after_coarsening(nx, ny, nz, coarsenLevel)

domain = mesh.create_box(
    comm=comm,
    points=[(0.0, 0.0, 0.0), (Lx, Ly, Lz)],
    n= n0, #[nx//coarsenLevel, ny//coarsenLevel, nz//coarsenLevel],
    cell_type=CellType.hexahedron,
    dtype=np.float64,  # Required for compatibility with FEM
    ghost_mode=GhostMode.shared_facet,
)
# Create connectivity for boundary facets
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim - 1, tdim)
update_peak_memory()

# --- Material Tagging (Local) ---
num_cells_local = domain.topology.index_map(tdim).size_local
local_cells = np.arange(num_cells_local, dtype=np.int32)
coords_local = mesh.compute_midpoints(domain, tdim, local_cells)
# Convert coordinates to float32 and ensure contiguous (memory optimization)
coords_local = np.ascontiguousarray(coords_local, dtype=np.float32)

material_tags_local = np.zeros(num_cells_local, dtype=np.int32)
for i, (x, y, z) in enumerate(coords_local):
    # Compute corresponding image indices (ensure within bounds)
    ix = np.clip(int(x / voxel_size), 0, nx - 1)
    iy = np.clip(int(y / voxel_size), 0, ny - 1)
    iz = np.clip(int(z / voxel_size), 0, nz - 1)
    material_tags_local[i] = image_buf[iz, iy, ix]
update_peak_memory()
# Free image buffer after tagging
comm.Barrier()  # synchronize before freeing distributed image
del image_buf
gc.collect()

cell_tags = meshtags(domain, tdim, local_cells, material_tags_local)

# --- Material Properties & FEM Setup ---
V0 = functionspace(domain, ("DG", 0))  # Discontinuous space for cell-wise constants
E_func = fem.Function(V0)
nu_func = fem.Function(V0)
# Use float32 for material property fields to save memory
for phase, (K, G) in materials.items():
    indices = cell_tags.indices[cell_tags.values == phase]
    if len(indices) > 0:
        # Compute E and nu for this phase (single precision)
        K32 = np.float32(K); G32 = np.float32(G)
        denom = 3.0 * K32 + G32
        if abs(denom) > 1e-12:
            E_val = 9.0 * K32 * G32 / denom
            nu_val = (3.0 * K32 - 2.0 * G32) / (2.0 * denom)
            # Assign values if they are physical
            if E_val > 0 and -1.0 < nu_val < 0.5:
                E_func.x.array[indices] = E_val
                nu_func.x.array[indices] = nu_val
            else:
                if comm.rank == 0:
                    print(f"Warning: Non-physical E ({E_val:.2e}) or nu ({nu_val:.2e}) for phase {phase}.", file=sys.stderr)
        else:
            if comm.rank == 0:
                print(f"Warning: Denominator (3K+G) near zero for phase {phase}.", file=sys.stderr)

# --- FEM Setup (Linear Elasticity) ---
V = functionspace(domain, ("CG", 1, (tdim,)))  # Vector function space for displacement

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
def epsilon(w): 
    return sym(nabla_grad(w))
def sigma(w, E, nu):
    # Stress tensor for isotropic linear elasticity
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    return 2.0 * mu * epsilon(w) + lmbda * tr(epsilon(w)) * Identity(tdim)

dx_measure = ufl.Measure("dx", domain=domain, metadata={'quadrature_degree': 2})
a = inner(sigma(u, E_func, nu_func), epsilon(v)) * dx_measure
zero_vec = np.zeros(tdim)
L = inner(Constant(domain, zero_vec), v) * dx_measure
# Use float32 constant for zero body force vector

# Compile forms in float32 precision for assembly (memory optimization)
a_form = fem.form(a)  # Default is float64
L_form = fem.form(L)

# --- Boundary Conditions and Solves ---
# Volumetric Strain BC (uniform expansion on all boundaries)
if comm.rank == 0:
    print("Setting up Volumetric BCs...")
    sys.stdout.flush()
boundary_facets_all = mesh.exterior_facet_indices(domain.topology)
boundary_dofs_all = fem.locate_dofs_topological(V, tdim - 1, boundary_facets_all)
# Convert to contiguous int32 array for efficiency
boundary_dofs_all = np.asarray(boundary_dofs_all, dtype=np.int32)

epsilon0 = np.float32(1e-4)  # Small applied strain
u_bc_vol = fem.Function(V)
u_bc_vol.interpolate(lambda x: epsilon0 * x)  # u = [ε*x, ε*y, ε*z] on boundaries
bc_vol = dirichletbc(u_bc_vol, boundary_dofs_all)

# Monitor memory usage before solving
print_memory_usage(comm, "Pre-solve memory")
update_peak_memory()
del coords_local, material_tags_local
gc.collect()
print_memory_usage(comm, "After cleanup")

# Solve for Bulk Modulus (K_eff) using single-precision linear system
solver_params_cleaned = {k: v for k, v in solver_params.items() if k != 'memory_optimized'}
problem_vol = LinearProblem(a_form, L_form, bcs=[bc_vol],
                            petsc_options=solver_params_cleaned,
                            u=fem.Function(V))
u_vol = problem_vol.solve()
# Using float32 for volumetric problem
print_memory_usage(comm, "Post volumetric solve")

# Clean up volumetric problem objects to free memory
del problem_vol, bc_vol, u_bc_vol
gc.collect()

# Compute effective bulk modulus K_eff
sigma_vol_expr = sigma(u_vol, E_func, nu_func)
trace_sigma_vol = tr(sigma_vol_expr)
vol_form = fem.form(1.0 * dx_measure)
trace_sigma_form = fem.form(trace_sigma_vol * dx_measure)
volume_local = fem.assemble_scalar(vol_form)
trace_sigma_local = fem.assemble_scalar(trace_sigma_form)
total_volume = comm.allreduce(volume_local, op=MPI.SUM)
total_trace_sigma = comm.allreduce(trace_sigma_local, op=MPI.SUM)
if total_volume < 1e-12 or abs(epsilon0) < 1e-12:
    K_eff = 0.0
    if comm.rank == 0:
        print("Warning: Total volume or applied strain is near zero. K_eff set to 0.")
else:
    K_eff = total_trace_sigma / (9 * epsilon0 * total_volume)
if comm.rank == 0:
    print(f"Volumetric solve complete. K_eff = {K_eff:.4f}")
    sys.stdout.flush()

# Uniaxial Tension BC (strain applied in X direction only)
if comm.rank == 0:
    print("Setting up Uniaxial BCs (X-dir)...")
    sys.stdout.flush()
# Identify facets on left (x=0) and right (x=Lx) boundaries
def left_boundary(x):
    return np.isclose(x[0], 0.0)
def right_boundary(x):
    return np.isclose(x[0], Lx)
left_facets = mesh.locate_entities_boundary(domain, tdim - 1, left_boundary)
right_facets = mesh.locate_entities_boundary(domain, tdim - 1, right_boundary)
# Locate DOFs for x-displacement on those facets (subspace V.sub(0))
left_dofs_x = fem.locate_dofs_topological(V.sub(0), tdim - 1, left_facets)
right_dofs_x = fem.locate_dofs_topological(V.sub(0), tdim - 1, right_facets)
# Convert dof index arrays to int32 (contiguous)
left_dofs_x = np.asarray(left_dofs_x, dtype=np.int32)
right_dofs_x = np.asarray(right_dofs_x, dtype=np.int32)
delta_x = epsilon0 * Lx  # Applied displacement on right face corresponding to strain
bc_left = dirichletbc(PETSc.ScalarType(0.0), left_dofs_x, V.sub(0))
bc_right = dirichletbc(PETSc.ScalarType(delta_x), right_dofs_x, V.sub(0))

# Solve for Young's Modulus (E_eff) using float32 linear system
if comm.rank == 0: 
    print("Solving Uniaxial problem...")
    sys.stdout.flush()
problem_uni = LinearProblem(a_form, L_form, bcs=[bc_left, bc_right],
                            petsc_options=solver_params_cleaned,
                            u=fem.Function(V))
u_uni = problem_uni.solve()

# Compute effective Young's modulus E_eff
sigma_uni_expr = sigma(u_uni, E_func, nu_func)
sigma_xx_uni = sigma_uni_expr[0, 0]
sigma_xx_form = fem.form(sigma_xx_uni * dx_measure)
sigma_xx_local = fem.assemble_scalar(sigma_xx_form)
total_sigma_xx = comm.allreduce(sigma_xx_local, op=MPI.SUM)
update_peak_memory()
if total_volume < 1e-12 or abs(epsilon0) < 1e-12:
    E_eff = 0.0
    if comm.rank == 0:
        print("Warning: Total volume or applied strain is near zero. E_eff set to 0.")
else:
    E_eff = (total_sigma_xx / total_volume) / epsilon0
if comm.rank == 0:
    print(f"Uniaxial solve complete. E_eff = {E_eff:.4f}")
    sys.stdout.flush()

# Clean up uniaxial problem and BCs to free memory
del problem_uni, bc_left, bc_right
gc.collect()
# Free material property functions now that solving is done
del E_func, nu_func
gc.collect()

# Calculate Effective Shear Modulus (G_eff) from K_eff and E_eff
G_eff = 0.0
if abs(E_eff) > 1e-12 and abs(K_eff) > 1e-12:
    denominator_nu = 6.0 * K_eff
    if abs(denominator_nu) > 1e-12:
        nu_eff = (3.0 * K_eff - E_eff) / denominator_nu
        if -1.0 < nu_eff < 0.5:
            denominator_G = 2.0 * (1.0 + nu_eff)
            if abs(denominator_G) > 1e-12:
                G_eff = E_eff / denominator_G
            elif comm.rank == 0:
                print(f"Warning: Denominator (2*(1+nu_eff)) is near zero ({denominator_G:.2e}). Cannot compute G_eff.", file=sys.stderr)
        elif comm.rank == 0:
            print(f"Warning: Derived effective Poisson's ratio ({nu_eff:.4f}) is non-physical. Cannot compute G_eff.", file=sys.stderr)
    elif comm.rank == 0:
        print(f"Warning: Denominator (6*K_eff) for nu_eff is near zero ({denominator_nu:.2e}). Cannot compute G_eff.", file=sys.stderr)
elif comm.rank == 0:
    print("Warning: Effective E or K is near zero. Cannot reliably compute G_eff.", file=sys.stderr)

# --- Final Output (printing results) ---
end_time = time.time()
# (Peak memory `peak_memory` and other outputs recorded as before)
if comm.rank == 0:
    print("\n--- Effective Properties ---")
    unit = "MPa"  # Adjust unit if input K,G given in different units
    print(f"Effective Bulk Modulus (K_eff): {K_eff:.4f} {unit}")
    print(f"Effective Young's Modulus (E_eff): {E_eff:.4f} {unit}")
    print(f"Effective Shear Modulus (G_eff): {G_eff:.4f} {unit} (Derived)")
    print(f"Effective Porosity: {exportPorosity*100.0:.2f}%")
    print(f"Fractal Dimension: {fractalDImension:.4f}")
    print(f"Total Execution time: {end_time - start_time:.2f} seconds")
    print("--------------------------")
    # Save results to file
    output_filename = f"modulus_output_{image_name.replace('.', '_')}_factor{downsample_factor}_{uniaxial_direction}.txt"
    try:
        with open(output_filename, 'w') as f:
            f.write(f"Image: {image_name}\n")
            f.write(f"Downsample Factor: {downsample_factor}\n")
            f.write(f"Original Voxel Size: {voxel_size_orig}\n")
            f.write(f"Mesh Voxel Size: {voxel_size}\n")
            f.write(f"Mesh Dimensions (Nx, Ny, Nz): ({nx}, {ny}, {nz})\n")
            f.write(f"Domain Size (Lx, Ly, Lz): ({Lx:.4f}, {Ly:.4f}, {Lz:.4f})\n")
            f.write(f"Applied Strain (epsilon0): {epsilon0:.2e}\n")
            f.write("---\n")
            f.write(f"K_eff: {K_eff:.4f} {unit}\n")
            f.write(f"E_eff: {E_eff:.4f} {unit}\n")
            f.write(f"G_eff: {G_eff:.4f} {unit}\n")
            f.write(f"Effective Porosity: {exportPorosity*100.0:.2f}%\n")
            f.write(f"Peak memory usage: {peak_memory:.2f} MB\n")
            f.write("---\n")
            f.write(f"Total Execution time: {end_time - start_time:.2f} seconds\n")
            f.write(f"Number of MPI processes: {comm.size}\n")
        print(f"Output saved to: {output_filename}")
    except Exception as e:
        print(f"Error writing output file '{output_filename}': {e}", file=sys.stderr)
