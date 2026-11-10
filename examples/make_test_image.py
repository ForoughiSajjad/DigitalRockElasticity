"""
Create small synthetic 3D test images for DigitalRockElasticity.

    python3 make_test_image.py            # writes homogeneous.tif and spheres.tif

homogeneous.tif  - a cube made entirely of solid (label 1).
                   With the affine boundary conditions used by the solver the exact
                   answer is the input moduli, so K_eff = K and G_eff = G.
                   Use it to check your installation (see README, "Sanity check").
spheres.tif      - a solid cube with random spherical pores (label 0), for a quick
                   demonstration with a non-trivial porous medium.

Images are 8-bit, shape (z, y, x), which is what the solver reads.
"""
import numpy as np
import tifffile

N = 40  # voxels per side

# 1. homogeneous solid
tifffile.imwrite("homogeneous.tif", np.ones((N, N, N), dtype=np.uint8))

# 2. solid with random spherical pores
rng = np.random.default_rng(0)
img = np.ones((N, N, N), dtype=np.uint8)
zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
for _ in range(25):
    c = rng.uniform(0, N, 3)
    r = rng.uniform(3, 6)
    img[(zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 < r ** 2] = 0
tifffile.imwrite("spheres.tif", img)
print(f"wrote homogeneous.tif and spheres.tif ({N}^3 voxels); "
      f"spheres porosity = {(img == 0).mean():.3f}")
