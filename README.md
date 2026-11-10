# DigitalRockElasticity

Compute the effective elastic moduli of porous materials directly from segmented 3D micro-CT images, using a voxel-based 3D linear-elasticity finite-element solver built with [FEniCS / DOLFINx](https://fenicsproject.org) and PETSc. It runs in parallel with MPI.

**Inputs:** a segmented 3D image (one integer label per voxel) and the bulk and shear modulus of each label.
**Outputs:** the effective bulk modulus *K*<sub>eff</sub>, shear modulus *G*<sub>eff</sub>, Young's modulus *E*<sub>eff</sub>, Poisson's ratio *ν*<sub>eff</sub>, plus the porosity and box-counting fractal dimension of the solid phase.

The code was written for the study of termite nest walls (see [Citation](#citation)), but it works for any segmented porous medium.

---

## Contents

| File | Purpose |
|---|---|
| `testRun9y.py` | The solver. |
| `input_y.dat` | Example input file (the default read by the solver). |
| `run9.sh` | Helper to run the solver in Docker with MPI. |
| `Dockerfile` | Builds an image with DOLFINx and the Python packages needed. |
| `examples/make_test_image.py` | Creates two small synthetic test images. |
| `examples/input_homogeneous.dat`, `examples/input_spheres.dat` | Input files for those test images. |
| `examples/check_output.py` | Checks a result against the exact answer for the homogeneous test. |

---

## Method in brief

1. The image is read as a 3D array and each voxel becomes one hexahedral element (or several voxels per element if you coarsen the mesh, see `coarsening_level`).
2. Each label is given an isotropic linear-elastic material from its bulk modulus *K* and shear modulus *G*: *E* = 9*KG*/(3*K*+*G*), *ν* = (3*K* − 2*G*)/(2(3*K*+*G*)).
3. Two small-strain problems are solved on the cube with **affine (kinematic) displacement boundary conditions** applied on all six faces:
   - **Uniform expansion**, *u* = *ε*<sub>0</sub>**x**. The effective bulk modulus is *K*<sub>eff</sub> = ⟨tr *σ*⟩ / (9 *ε*<sub>0</sub>).
   - **Simple shear in the x–y plane**, *u* = (½*ε*<sub>0</sub>*y*, ½*ε*<sub>0</sub>*x*, 0). The effective shear modulus is *G*<sub>eff</sub> = ⟨*σ*<sub>xy</sub>⟩ / *ε*<sub>0</sub>.

   Here ⟨·⟩ is the volume average over the cube, and *ε*<sub>0</sub> = 10<sup>-4</sup>.
4. *E*<sub>eff</sub> and *ν*<sub>eff</sub> are then computed from *K*<sub>eff</sub> and *G*<sub>eff</sub> using the isotropic relations above. They are reference values, not independent measurements.

The linear systems are solved with conjugate gradients and algebraic multigrid (hypre BoomerAMG), relative tolerance 10<sup>-6</sup>.

---

## Installation

### Option 1: Docker (recommended)

```bash
docker build -t digitalrock-elasticity .
```

The `Dockerfile` starts from the official `dolfinx/dolfinx` image and adds `scikit-image`, `tifffile` and `psutil`.

### Option 2: your own DOLFINx environment

You need Python 3 with:

- DOLFINx and PETSc with hypre, built with MPI (the script uses the DOLFINx 0.8/0.9 API: `dolfinx.fem.functionspace`, `dolfinx.fem.petsc.LinearProblem`. Newer DOLFINx releases changed `LinearProblem`, so pin the version),
- `mpi4py`, `petsc4py`, `numpy`,
- `scikit-image`, `tifffile`, `psutil`.

---

## Quick start: sanity check (about a minute)

A homogeneous solid with affine boundary conditions has an exact solution, so *K*<sub>eff</sub> and *G*<sub>eff</sub> must equal the input values. This checks the installation end to end.

```bash
cd examples
python3 make_test_image.py            # writes homogeneous.tif and spheres.tif
cd ..
cp examples/homogeneous.tif examples/input_homogeneous.dat .

# with Docker (4 MPI ranks)
docker run --rm -v "$(pwd)":/workspace -w /workspace digitalrock-elasticity \
    mpirun -n 4 python3 /opt/digitalrock/testRun9y.py input_homogeneous.dat

python3 examples/check_output.py modulus_output_homogeneous_tif_factor1.0_x.txt 37 44
```

Expected result for the input moduli *K* = 37 and *G* = 44 (quartz, in GPa):

```
K_eff: 37.0000 GPa
G_eff: 44.0000 GPa
E_eff: 94.5290 GPa
nu_eff: 0.0742
```

For a porous test, repeat with `spheres.tif` and `input_spheres.dat` (a solid cube with random spherical pores, about 14% porosity). *K*<sub>eff</sub> and *G*<sub>eff</sub> must come out below 37 and 44.

---

## Preparing the image

- **Format:** a 3D TIFF stack readable by `skimage.io.imread`, array shape `(z, y, x)`.
- **Labels:** integer phase labels, one per voxel (for example 0 = pore, 1 = grain). The solver converts the image to 8-bit, so labels must be integers from 0 to 255.
- **Every label in the image needs a row in the material table** (see below). A label without a row gets zero stiffness.
- **Cubic or cuboid** domains are both fine. Voxels are assumed cubic.

---

## Input file format

The input file is a plain text file. The first five lines must appear **in this order** and no comment lines are allowed among them. The material table follows.

```text
image_name = "subvolume.tif"
uniaxial_direction = "x"
voxel_size = 5
coarsening_level = 1
pore_label = 0
phase   K	G
0	0.0001	0.0001
1	5.0	15.0
```

| Line | Meaning |
|---|---|
| `image_name` | Path of the TIFF file, relative to the working directory. |
| `uniaxial_direction` | `x`, `y` or `z`. Reorients the image before solving: `x` uses the array as read, `y` and `z` transpose its axes. The shear test is always done in the solver's x–y plane, so the three settings probe different planes. *K*<sub>eff</sub> should be the same for all three. |
| `voxel_size` | Voxel edge length, in any length unit. It only sets the size of the domain. The moduli do not depend on it. |
| `coarsening_level` | Mesh coarsening factor. The number of elements along each axis is the voxel count divided by this factor (rounded, at least 2). `1` means one element per voxel. `2` gives half as many elements per axis, about 8× fewer in total. Non-integer values are accepted. With a factor above 1, each element takes the label of the voxel at its centre. The labels are not averaged, so coarsening changes the geometry slightly. |
| `pore_label` | The label of the pore phase. It is used to compute the porosity and the fractal dimension of the solid. |
| header line | `phase K G` (any whitespace). |
| material rows | One row per label: `label  K  G`, separated by whitespace. Blank lines and lines starting with `#` are skipped. |

**Units:** moduli can be given in any unit. The results come out in the same unit (the output files say "GPa" because that is the unit used for the examples).

**The pore phase is not empty.** Pore voxels are given a very small stiffness (here 10<sup>-4</sup>, versus 5 to 37 for the solid) so that the stiffness matrix stays non-singular. The default of 10<sup>-4</sup> is about 5 orders of magnitude below the solid. Keep the pore stiffness at least that far below the solid, but not zero.

Phases whose *K* and *G* give a non-physical *E* or *ν* are skipped with a warning and get zero stiffness.

By default the solver reads `input_y.dat` from the current directory. You can pass a different file as the first argument: `python3 testRun9y.py my_input.dat`.

---

## Running

With Docker and the helper script (default: `input_y.dat`, 10 MPI ranks):

```bash
./run9.sh                          # input_y.dat, 10 ranks
./run9.sh my_input.dat 32          # your input file, 32 ranks
```

Without Docker:

```bash
mpirun -n 10 python3 testRun9y.py my_input.dat
```

### Size and cost

The number of degrees of freedom is 3 × (*n*<sub>x</sub>+1)(*n*<sub>y</sub>+1)(*n*<sub>z</sub>+1), where *n*<sub>i</sub> is the number of elements along each axis (the voxel count divided by `coarsening_level`). Memory and run time grow roughly linearly with this number. Two things to know:

- Rank 0 reads the whole image and broadcasts it, so **every MPI rank holds a copy of the full 8-bit image** while the mesh is built.
- Start with a small subvolume or a higher `coarsening_level`, check the result, then increase the resolution.

---

## Output

The solver prints progress, memory use and the final results, and writes a file named

```
modulus_output_<image name with '.' replaced by '_'>_factor<coarsening_level>_<direction>.txt
```

for example `modulus_output_subvolume_tif_factor1.0_x.txt`. It contains:

```
Image: subvolume.tif
...
K_eff: ...      (bulk modulus)
E_eff: ...      (Young's modulus, from K_eff and G_eff)
G_eff: ...      (shear modulus)
nu_eff: ...     (Poisson's ratio)
Volumetric KSP reason: <code>, iterations: <n>
Shear KSP reason: <code>, iterations: <n>
Effective Porosity: ...%
Peak memory usage: ... MB
Total Execution time: ... seconds
Number of MPI processes: ...
```

**Check convergence.** A KSP reason greater than 0 means the linear solve converged. A value of 0 or below means it did not, and the results should not be used. In that case the solver also prints a warning to stderr.

The porosity is the fraction of voxels with the pore label. The fractal dimension (printed on screen) is the box-counting dimension of the solid phase.

---

## Assumptions and limitations

Read these before interpreting the numbers.

- **Upper-bound-type estimates.** The affine displacement boundary conditions are stiffer than the true response of a finite sample, so *K*<sub>eff</sub> and *G*<sub>eff</sub> are upper bounds on the moduli of the medium. The overestimate shrinks as the subvolume grows.
- **Perfect bonding.** Neighbouring solid voxels are fully bonded wherever they touch. There is no contact mechanics, no friction and no cement model. Where grains touch at only a few voxels, the contact is therefore stiffer than a real grain contact.
- **Only the resolved structure.** Porosity or cracks below the image resolution are not represented. If you want to include them, you must reduce the phase moduli yourself.
- **Linear, small strain, isotropic phases.** Each phase is isotropic. The cube-level response is computed in one shear plane only, and *E*<sub>eff</sub> and *ν*<sub>eff</sub> assume isotropy at the cube level.
- **Dry medium.** The pore phase carries no fluid, so there is no poroelastic or saturation effect.
- **Hard-coded settings.** The applied strain (10<sup>-4</sup>), the solver tolerances and the `downsample_factor` (a voxel-averaging option, default 1 = off) are set at the top of `testRun9y.py`.

---

## Troubleshooting

| Message | Cause |
|---|---|
| `<something> must specify <name>` (for example `Second line must specify coarsening_level`) | One of the first five lines of the input file is missing or out of order. Compare with the example above. |
| `Input parsing failed on rank 0` | The same, or the file was not found. The exact reason is printed just above this line. |
| `Image error: ...` | The TIFF could not be read. Check `image_name` and the working directory (inside Docker, the image must be under the mounted folder). |
| `Non-physical E ... for phase ...` | The *K*, *G* pair you gave gives a non-physical *E* or *ν*. The phase gets zero stiffness. Fix the moduli. |
| `NOT CONVERGED` | The linear solve failed. Check that every label has a material row and that the pore stiffness is not exactly zero. |
| Out of memory | Raise `coarsening_level`, use a smaller subvolume, or use more nodes. |

---

## Citation

If you use this code, please cite the paper it was developed for:

```bibtex
@unpublished{foroughi2026termite,
  title  = {How termite nest walls balance gas exchange, strength, moisture retention and thermal stability},
  author = {Foroughi, Sajjad and Naderi, Sadjad and Suo, Si and Bijeljic, Branko and Blunt, Martin J.},
  note   = {Submitted},
  year   = {2026}
}
```

Replace this entry with the journal reference once the paper is published.

The pore-network extraction and flow codes used in the same study are in [ImperialCollegeLondon/porescale](https://github.com/ImperialCollegeLondon/porescale).
